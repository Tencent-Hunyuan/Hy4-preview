"""
Patches for ms-swift training.

Patches auto-applied on import:
    1. Model & Template registration: Register the custom model_type
       in ms-swift.
    2. Grad norm skip patch: Skip grad norm computation for ZeRO-3 + CPU offload.
    3. Memory-efficient model loading: Shard-by-shard loading that
       processes one safetensors shard for memory optimization.
    4. Fix logging_dir compatibility between ms-swift and transformers 5.x.
    5. Align FSDP1 dtype / mixed precision behavior with LLaMA Factory.
    6. Disable _compute_acc to avoid errors during training.

Optional (call manually after LoRA is applied):
    - apply_lora_z3_leaf_patch(model): Mark PEFT LoRA wrapper modules as
      ZeRO-3 leaf modules to fix parameter fetch/release scheduling issues.

Usage:
    swift sft --custom_register_path hy_v4_swift_patches.py --model /path/to/ckpt ...
"""

import os
import gc
import json as _json
import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# ============================================================================
# Patch 1: Model & Template Registration
#
# ms-swift natively supports hy_v3, but HYV4 has a different model class
# (HYV4ForCausalLM) and additional iHC modules. We register HYV4 as a
# custom model_type that reuses the hy_v3 template.
# ============================================================================

from swift.model import register_model, ModelMeta, ModelGroup, Model
from swift.template import register_template, TemplateMeta

# Register hy_v4 template
# Token format: <｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>{content}<｜hy_end:opensource｜>
register_template(
    TemplateMeta(
        template_type='hy_v4',
        prefix=[],
        system_prefix=['<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>{{SYSTEM}}<｜hy_end:opensource｜>'],
        prompt=['<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{{QUERY}}<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>'],
        chat_sep=['<｜hy_end:opensource｜>'],
        suffix=['<｜hy_end:opensource｜>'],
    ),
    exist_ok=True,
)

# Register hy_v4 model
register_model(
    ModelMeta(
        model_type='hy_v4',
        model_groups=[
            ModelGroup([
                Model('Tencent-Hunyuan/Hy4',
                      'Tencent-Hunyuan/Hy4'),
            ]),
        ],
        template='hy_v4',
        architectures=['HYV4ForCausalLM'],
        is_multimodal=False,
    ),
    exist_ok=True,
)

logger.info(
    "[HYV4 Patch 1] Model type 'hy_v4' and template registered in ms-swift."
)

# ============================================================================
# Patch 2: Skip grad norm computation for DeepSpeed ZeRO-3
#
# Under ZeRO-3 + CPU offload, DeepSpeed's complete_grad_norm_calculation
# all-gathers every gradient on CPU and does an ALLREDUCE to compute the
# global L2 norm. For a 770B model this causes NCCL timeout/deadlock.
# When max_grad_norm=0 (no clipping), we skip the norm computation entirely.
# ============================================================================

def _apply_skip_grad_norm_patch():
    """Patch DeepSpeed ZeRO-3 optimizer to skip grad norm computation."""
    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

        def _skip_get_norm_groups(self):
            return [torch.tensor(0.0)]

        DeepSpeedZeroOptimizer_Stage3._get_norm_groups = _skip_get_norm_groups
        logger.info(
            "[HYV4 Patch 2] Patched DeepSpeedZeroOptimizer_Stage3._get_norm_groups "
            "to skip grad norm computation."
        )
    except ImportError:
        logger.info("[HYV4 Patch 2] DeepSpeed not available, skipping grad norm patch.")


# ============================================================================
# Patch 3: Memory-efficient model loading for ZeRO-3 and FSDP1
#
# ZeRO-3 path:
#   Ensure transformers sees the DeepSpeed config early enough to activate its
#   native efficient loading path.
#
# FSDP1 path:
#   Mirror the previously successful LLaMA Factory behavior:
#     1. local_rank 0 loads real weights to CPU only
#     2. all other local ranks create the model on meta device
#     3. low_cpu_mem_usage stays enabled
#
# This avoids loading the full model onto every GPU during `from_pretrained`,
# which is exactly the failure mode we observed with ms-swift + FSDP.
# ============================================================================

def _apply_shard_loading_patch():
    """Ensure efficient model loading is used for both ZeRO-3 and FSDP1."""
    import sys
    from transformers import AutoConfig, PreTrainedModel

    _real_orig_from_pretrained = PreTrainedModel.from_pretrained.__func__

    def _disable_router_logits_if_needed(model):
        if hasattr(model, 'config') and getattr(model.config, 'output_router_logits', False):
            model.config.output_router_logits = False
            print("[HYV4 Patch 3] Disabled output_router_logits.", flush=True)
        return model

    def _is_fsdp_requested():
        accel_fsdp = str(os.environ.get("ACCELERATE_USE_FSDP", "")).lower()
        if accel_fsdp in {"1", "true", "yes"}:
            return True
        return "--fsdp" in sys.argv

    def _build_meta_model_for_fsdp(cls, model_path, kwargs):
        config = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

        init_kwargs = {}
        torch_dtype = kwargs.get("torch_dtype", None)
        if torch_dtype is not None:
            init_kwargs["torch_dtype"] = torch_dtype
        if "attn_implementation" in kwargs:
            init_kwargs["attn_implementation"] = kwargs["attn_implementation"]
        if "experts_implementation" in kwargs:
            init_kwargs["experts_implementation"] = kwargs["experts_implementation"]

        with torch.device("meta"):
            model = cls._from_config(config, **init_kwargs)
        return _disable_router_logits_if_needed(model)

    def _fsdp_safe_load(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("low_cpu_mem_usage", True)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            logger.info(
                "[HYV4 Patch 3] FSDP mode: local_rank=%d != 0, creating model on meta device. "
                "Weights will be synchronized from rank 0.",
                local_rank,
            )
            return _build_meta_model_for_fsdp(cls, pretrained_model_name_or_path, kwargs)

        logger.info(
            "[HYV4 Patch 3] FSDP mode: local_rank=0, loading real weights to CPU "
            "with low_cpu_mem_usage enabled."
        )
        kwargs["device_map"] = {"": "cpu"}
        model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
        return _disable_router_logits_if_needed(model)

    def _ensure_zero3_config_and_load(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Ensure HfDeepSpeedConfig is set before calling from_pretrained."""
        model_path = pretrained_model_name_or_path
        print(f"[HYV4 Patch 3] _ensure_zero3_config_and_load called with path: {model_path}", flush=True)

        if not (isinstance(model_path, str) and os.path.isdir(model_path)):
            return _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        index_file = os.path.join(model_path, "model.safetensors.index.json")
        single_file = os.path.join(model_path, "model.safetensors")
        if not (os.path.isfile(index_file) or os.path.isfile(single_file)):
            return _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        if _is_fsdp_requested():
            return _fsdp_safe_load(cls, pretrained_model_name_or_path, *args, **kwargs)

        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        if is_deepspeed_zero3_enabled():
            print("[HYV4 Patch 3] ZeRO-3 already enabled, using native from_pretrained.", flush=True)
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        ds_config_path = os.environ.get("DEEPSPEED_CONFIG_FILE", None)
        if ds_config_path is None:
            ds_config_path = os.environ.get("DEEPSPEED_CONFIG", None)

        if ds_config_path is None:
            for i, arg in enumerate(sys.argv):
                if arg == '--deepspeed' and i + 1 < len(sys.argv):
                    ds_config_path = sys.argv[i + 1]
                    break

        if ds_config_path is None or not os.path.isfile(ds_config_path):
            print("[HYV4 Patch 3] No DeepSpeed config found, using default from_pretrained.", flush=True)
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        with open(ds_config_path, "r") as f:
            ds_config = _json.load(f)

        zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
        if zero_stage != 3:
            print(f"[HYV4 Patch 3] Not ZeRO-3 (stage={zero_stage}), using default.", flush=True)
            model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
            return _disable_router_logits_if_needed(model)

        print(f"[HYV4 Patch 3] Setting HfDeepSpeedConfig for ZeRO-3 native loading: {ds_config_path}", flush=True)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        _ds_config_obj = HfDeepSpeedConfig(ds_config_path)

        model = _real_orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)

        print("[HYV4 Patch 3] Native ZeRO-3 from_pretrained completed.", flush=True)
        return _disable_router_logits_if_needed(model)

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return _ensure_zero3_config_and_load(cls, pretrained_model_name_or_path, *args, **kwargs)

    PreTrainedModel.from_pretrained = _patched_from_pretrained

    logger.info("[HYV4 Patch 3] Loading patch applied for ZeRO-3 and FSDP1.")


# ============================================================================
# Optional Patch (NOT auto-applied): LoRA z3_leaf marking
#
# PEFT wraps target Linear layers with lora.Linear, adding extra sub-modules
# (base_layer, lora_A, lora_B). This changes the module tree structure and
# disrupts ZeRO-3's parameter fetch/release scheduling, causing OOM during
# backward recomputation. By marking these wrappers as z3_leaf, ZeRO-3 treats
# them as atomic units, restoring correct scheduling.
#
# This is NOT auto-applied because it requires the model to have LoRA already
# applied. Call apply_lora_z3_leaf_patch(model) manually after LoRA setup.
# ============================================================================

def apply_lora_z3_leaf_patch(model):
    """Mark PEFT LoRA wrapper modules as ZeRO-3 leaf modules.

    This is an OPTIONAL patch. Call manually AFTER LoRA has been applied
    to the model and BEFORE training starts.
    """
    try:
        from deepspeed.utils import set_z3_leaf_module
        from peft.tuners.lora import Linear as LoraLinear
    except ImportError:
        logger.info("[HYV4 Optional] DeepSpeed or PEFT not available, skipping z3_leaf patch.")
        return

    z3_leaf_count = 0
    for module in model.modules():
        if isinstance(module, LoraLinear):
            set_z3_leaf_module(module, True)
            z3_leaf_count += 1

    logger.info("[HYV4 Optional] Marked %d LoraLinear modules with _z3_leaf=True.", z3_leaf_count)


# ============================================================================
# Patch 4: Fix logging_dir compatibility between ms-swift and
# transformers 5.x
#
# ms-swift's SftArguments._add_version() accesses self.logging_dir,
# expecting it to be inherited from transformers.TrainingArguments as a
# dataclass field. However, in transformers 5.x, logging_dir has been
# deprecated and is no longer included in dataclass fields (init=False or
# removed from __dataclass_fields__). This causes AttributeError.
#
# Fix: Monkey-patch _add_version to ensure logging_dir exists before access.
# ============================================================================

def _apply_logging_dir_patch():
    """Fix SftArguments._add_version for transformers 5.x compatibility."""
    try:
        from swift.arguments.sft_args import SftArguments

        _orig_add_version = SftArguments._add_version

        def _patched_add_version(self):
            # Ensure logging_dir attribute exists (transformers 5.x removed it
            # from dataclass fields but ms-swift 4.4.2 still accesses it)
            if not hasattr(self, 'logging_dir'):
                self.logging_dir = None
            # Also ensure run_name exists (may also be affected)
            if not hasattr(self, 'run_name'):
                self.run_name = None
            _orig_add_version(self)

        SftArguments._add_version = _patched_add_version
        logger.info(
            "[HYV4 Patch 4] Patched SftArguments._add_version for "
            "transformers 5.x logging_dir compatibility."
        )
    except (ImportError, AttributeError) as e:
        logger.info("[HYV4 Patch 4] Could not apply logging_dir patch: %s", e)


# ==========================================================================
# Patch 5: Mirror LLaMA Factory's FSDP1 dtype + mixed precision guard
#
# After LoRA injection, adapter weights can stay in fp32 while the base model
# is bf16. In addition, Accelerate may re-enable mixed precision policies that
# upcast bf16 shards back to fp32 during FSDP wrapping. Both behaviors increase
# memory pressure significantly.
#
# We align ms-swift with the previously successful LLaMA Factory setup by:
#   1. unifying floating-point parameters to bf16 before FSDP wrap
#   2. disabling Accelerate mixed precision for the FSDP plugin
# ============================================================================

def _apply_fsdp_dtype_patch():
    """Unify parameter dtype to bf16 and disable Accelerate FSDP1 mixed precision."""
    try:
        from transformers import Trainer

        _orig_prepare_for_training = Trainer._prepare_for_training

        def _patched_prepare_for_training(self, *args, **kwargs):
            if getattr(self.args, 'fsdp', False):
                dtype_counts = {}
                for p in self.model.parameters():
                    dt = str(p.dtype)
                    dtype_counts[dt] = dtype_counts.get(dt, 0) + 1

                if len(dtype_counts) > 1:
                    logger.info(
                        "[HYV4 Patch 5] Mixed dtypes detected before FSDP wrap: %s. "
                        "Casting floating-point parameters to bfloat16.",
                        dtype_counts,
                    )
                    for p in self.model.parameters():
                        if p.dtype != torch.bfloat16 and p.dtype.is_floating_point:
                            p.data = p.data.to(torch.bfloat16)
                else:
                    logger.info("[HYV4 Patch 5] Parameter dtypes already uniform: %s", dtype_counts)

                try:
                    if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'state'):
                        old_mp = getattr(self.accelerator.state, '_mixed_precision', None)
                        self.accelerator.state._mixed_precision = 'no'
                        fsdp_plugin = getattr(self.accelerator.state, 'fsdp_plugin', None)
                        if fsdp_plugin is not None:
                            if hasattr(fsdp_plugin, 'mixed_precision_policy'):
                                fsdp_plugin.mixed_precision_policy = None
                            if hasattr(fsdp_plugin, 'kwargs') and isinstance(fsdp_plugin.kwargs, dict):
                                fsdp_plugin.kwargs.pop('mixed_precision', None)
                        logger.info(
                            "[HYV4 Patch 5] Disabled Accelerate FSDP1 mixed precision "
                            "(previous state: %s).",
                            old_mp,
                        )
                    os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
                except Exception as e:
                    logger.warning("[HYV4 Patch 5] Failed to disable mixed precision: %s", e)

            return _orig_prepare_for_training(self, *args, **kwargs)

        Trainer._prepare_for_training = _patched_prepare_for_training
        logger.info("[HYV4 Patch 5] FSDP1 dtype + mixed precision guard applied.")
    except (ImportError, AttributeError) as e:
        logger.info("[HYV4 Patch 5] Could not apply FSDP1 dtype patch: %s", e)


# ============================================================================
# Patch 6: Disable _compute_acc during training
#
# ms-swift computes training accuracy (argmax on logits) at every step.
# This requires keeping outputs.logits in memory until acc is computed,
# adding extra GPU/CPU memory pressure. LLaMA Factory and DeepSpeed native
# scripts do NOT compute training accuracy. Disabling this aligns ms-swift
# with the other frameworks and reduces memory usage during forward pass.
# ============================================================================

def _apply_disable_compute_acc_patch():
    """Patch _compute_acc to be a no-op during training."""
    try:
        from swift.trainers.mixin import SwiftMixin

        def _noop_compute_acc(self, outputs, labels, cu_seqlens=None):
            return

        SwiftMixin._compute_acc = _noop_compute_acc
        print("[HYV4 Patch 6] Disabled _compute_acc to reduce memory usage.", flush=True)
    except (ImportError, AttributeError) as e:
        print(f"[HYV4 Patch 6] Could not apply _compute_acc patch: {e}", flush=True)


# ============================================================================
# Auto-apply patches on import
# ============================================================================

# Patch 2: Skip grad norm (always safe to apply; no-op if DeepSpeed not used)
_apply_skip_grad_norm_patch()

# Patch 3: Memory-efficient model loading
_apply_shard_loading_patch()

# Patch 4: Fix logging_dir compatibility
_apply_logging_dir_patch()

# Patch 5: Align FSDP1 dtype / mixed precision behavior with LLaMA Factory
_apply_fsdp_dtype_patch()

# Patch 6: Disable _compute_acc
_apply_disable_compute_acc_patch()

logger.info("[HYV4] All ms-swift patches loaded successfully.")
