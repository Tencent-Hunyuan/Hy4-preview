# Copyright 2024 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import shutil
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.modeling_utils import unwrap_model


def print_args(args, name='arguments'):
    """Print arguments."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f'------------------------ {name} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {name} ---------------------', flush=True)


@dataclass
class ModelArguments:
    use_flash_attn: bool = field(
        default=False, 
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable Lora for faster training."})
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})


@dataclass
class DataArguments:
    train_data_file: str = field(default=None, metadata={"help": "Path to the training data."})
    max_seq_length: int = field(
        default=2048, 
        metadata={"help": "The max sequence length of the model inputs after tokenization."}
    )
    complex_data: Optional[str] = field(default=None)
    use_dummy_data: bool = field(default=False, metadata={"help": "Use dummy data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    min_lr: float = field(
        default=0.01, 
        metadata={"help": "The final learning rate at the end of the decay will be learning_rate * min_lr"}
    )


IGNORE_INDEX = -100

HY_START_ID = 120000
HY_MIDDLE_ID = 120001
HY_END_ID = 120025


class DummyDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length=512, length=1000):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        tokens = torch.randint(0, self.tokenizer.vocab_size, (self.max_seq_length, ))
        return {'input_ids': tokens, 'labels': tokens}


class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_length = 2048, prompt_format = 'mplus'):
        self.tokenizer = tokenizer
        self.prompt_format = prompt_format
        self.max_seq_length = max_seq_length

        self.data_list = self.load_data(data_file)

        # Pre-compute special token IDs for loss masking (Scheme B)
        self.hy_start_id = HY_START_ID
        self.hy_middle_id = HY_MIDDLE_ID
        self.hy_end_id = HY_END_ID
        # "assistant" is encoded as BPE tokens [611, 10372] ("ass" + "istant")
        self.assistant_bpe_ids = tokenizer.encode('assistant', add_special_tokens=False)
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data_list)

    def load_data(self, data_file):
        logging.info('Loading data: {}'.format(data_file))
        with open(data_file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logging.info("there are {} data in dataset".format(len(data_list)))
        return data_list

    def _find_assistant_turn_boundaries(self, token_ids):
        """Find assistant turn boundaries using Scheme B.
        
        Locates assistant turns by finding <hy_start> + "assistant" BPE tokens pattern,
        then marks loss from the <hy_middle> token (exclusive) to <hy_end> (inclusive).
        
        Returns:
            List of (start, end) tuples where:
            - start: position of <hy_middle> in the assistant turn (loss starts at start+1)
            - end: position of <hy_end> for that turn (loss includes this position)
        """
        boundaries = []
        assistant_len = len(self.assistant_bpe_ids)
        ids_list = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        n = len(ids_list)
        
        i = 0
        while i < n:
            # Look for <hy_start> token
            if ids_list[i] == self.hy_start_id:
                # Check if followed by "assistant" BPE tokens
                role_start = i + 1
                role_end = role_start + assistant_len
                if role_end <= n and ids_list[role_start:role_end] == self.assistant_bpe_ids:
                    # Check if followed by <hy_middle>
                    if role_end < n and ids_list[role_end] == self.hy_middle_id:
                        middle_pos = role_end
                        # Find the corresponding <hy_end>
                        end_pos = None
                        for j in range(middle_pos + 1, n):
                            if ids_list[j] == self.hy_end_id:
                                end_pos = j
                                break
                        if end_pos is not None:
                            boundaries.append((middle_pos, end_pos))
                            i = end_pos + 1
                            continue
            i += 1
        
        return boundaries

    def encode_data(self, data_dict):
        model_inputs = {}
        reasoning_effort = data_dict.get('reasoning_effort', None)
        if reasoning_effort is None:
            reasoning_effort = 'no_think'
        try:
            template_output = self.tokenizer.apply_chat_template(
                data_dict['messages'], tokenize=True, return_dict=False,
                reasoning_effort=reasoning_effort
            )
        except Exception as e:
            print(f"[ERROR] apply_chat_template failed: {e}")
            print(f"[ERROR] messages: {data_dict['messages']}")
            print(f"[ERROR] reasoning_effort: {reasoning_effort}")
            template_output = []
        
        # Debug: Check template_output type and content
        if isinstance(template_output, bool):
            print(f"[WARNING] apply_chat_template returned bool: {template_output}")
            print(f"[WARNING] messages: {data_dict['messages']}")
            print(f"[WARNING] reasoning_effort: {reasoning_effort}")
            template_output = []
        
        if isinstance(template_output, list) and len(template_output) > 0 and isinstance(template_output[0], list):
            template_output = template_output[0]
        
        # Ensure template_output is a list of integers
        if not isinstance(template_output, list) or not all(isinstance(x, int) for x in template_output):
            print(f"[WARNING] Invalid template_output format: {type(template_output)}, content: {template_output}")
            print(f"[WARNING] messages: {data_dict['messages']}")
            template_output = []
        
        message_tokens = torch.tensor(template_output, dtype=torch.long)

        # Handle empty message_tokens case
        if message_tokens.numel() == 0:
            print(f"[WARNING] Empty message_tokens, skipping data sample")
            input_ids = torch.tensor([], dtype=torch.long)
            labels = torch.tensor([], dtype=torch.long)
            attention_mask = torch.tensor([], dtype=torch.bool)
        else:
            # Scheme B: Find assistant turn boundaries and build labels
            boundaries = self._find_assistant_turn_boundaries(message_tokens)
            message_labels = torch.full_like(message_tokens, IGNORE_INDEX)
            
            for middle_pos, end_pos in boundaries:
                # Compute loss from the token after <hy_middle> to <hy_end> (inclusive)
                message_labels[middle_pos + 1:end_pos + 1] = message_tokens[middle_pos + 1:end_pos + 1]
            
            input_ids = message_tokens.to(torch.long)
            labels = message_labels.to(torch.long)

            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = input_ids.ne(self.pad_token_id).to(torch.bool)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask
        model_inputs["labels"] = labels

        return model_inputs

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        model_inputs = self.encode_data(data)
        
        # Check if the encoded data is empty (due to tokenization failure)
        if model_inputs["input_ids"].numel() == 0:
            # Return a valid placeholder sample to avoid crash
            eos_token_id = self.hy_end_id
            pad_token_id = self.pad_token_id
            
            # Create a minimal valid sequence
            placeholder_tokens = [self.hy_start_id, eos_token_id]
            placeholder_tokens = placeholder_tokens[:self.max_seq_length]
            
            input_ids = torch.tensor(placeholder_tokens, dtype=torch.long)
            labels = torch.tensor([IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long)[:self.max_seq_length]
            attention_mask = torch.tensor([1, 1], dtype=torch.bool)[:self.max_seq_length]
            
            # Pad to max_seq_length if needed
            if len(placeholder_tokens) < self.max_seq_length:
                padding_length = self.max_seq_length - len(placeholder_tokens)
                input_ids = torch.cat([input_ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((padding_length,), IGNORE_INDEX, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.bool)])
            
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        return model_inputs


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )


def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.use_dummy_data:
        train_dataset = DummyDataset(tokenizer, data_args.max_seq_length)
    else:
        train_dataset = SFTDataset(
            tokenizer=tokenizer, 
            data_file=data_args.train_data_file, 
            max_seq_length=data_args.max_seq_length
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# Copy tokenizer and config files when saving checkpoints
class CustomSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            # Copy tokenizer and config files to checkpoint directory
            tokenizer_files = [
                'config.json',
                'generation_config.json',
                'tokenizer_config.json',
                'tokenizer.json',
                'chat_template.jinja',
                'preprocessor_config.json',
                'hy.tiktoken',
                'tokenization_hy.py',
                'special_tokens_map.json',
            ]
            for fname in tokenizer_files:
                src = os.path.join(args.tokenizer_name_or_path, fname)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(output_dir, fname))

        return control


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print_args(model_args, 'model arguments')
    print_args(data_args, 'data arguments')
    print_args(training_args, 'training arguments')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.tokenizer_name_or_path,
        trust_remote_code = True
    )

    init_kwargs = {}
    if model_args.use_flash_attn:
        init_kwargs["attn_implementation"] = "flash_attention_2"
        # Workaround: transformers >= 5.x uses importlib.metadata.packages_distributions()
        # to verify flash-attn package name, which fails when the package is installed under
        # a custom distribution name (e.g. ptm-flash-attn). Patch the check to skip it.
        try:
            from transformers.modeling_flash_attention_utils import FLASH_ATTENTION_COMPATIBILITY_MATRIX
            _orig_pkg_check = FLASH_ATTENTION_COMPATIBILITY_MATRIX[2]["pkg_availability_check"]
            FLASH_ATTENTION_COMPATIBILITY_MATRIX[2]["pkg_availability_check"] = lambda *a, **kw: True
            print("[Patch] Bypassed flash_attn package distribution name check for FA2.")
        except Exception as e:
            print(f"[Patch] Could not patch FA2 pkg check (non-fatal): {e}")

    # Determine torch dtype
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # -----------------------------------------------------------------------
    # DeepSpeed ZeRO-3: Tell transformers that we are using ZeRO-3 so that
    # from_pretrained will shard the model across ranks instead of loading
    # the full model on each node's CPU (which would OOM for large models).
    #
    # NOTE: The ds_config may contain "auto" for batch size fields, which
    # DeepSpeed cannot parse at this stage (before Trainer resolves them).
    # We must fill in concrete values before passing to HfDeepSpeedConfig.
    # -----------------------------------------------------------------------
    if training_args.deepspeed:
        from transformers.integrations import HfDeepSpeedConfig

        # Load ds_config and resolve "auto" batch size fields
        ds_config = training_args.deepspeed
        if isinstance(ds_config, str):
            with open(ds_config, 'r') as f:
                ds_config = json.load(f)

        # Fill in batch size fields that DeepSpeed needs for zero.Init
        if ds_config.get("train_micro_batch_size_per_gpu", "auto") == "auto":
            ds_config["train_micro_batch_size_per_gpu"] = training_args.per_device_train_batch_size
        if ds_config.get("gradient_accumulation_steps", "auto") == "auto":
            ds_config["gradient_accumulation_steps"] = training_args.gradient_accumulation_steps
        if ds_config.get("train_batch_size", "auto") == "auto":
            ds_config["train_batch_size"] = (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
            )

        dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841 - must keep ref to avoid GC

    # Check if model weights exist (not just the directory)
    _has_weights = (
        training_args.model_name_or_path is not None
        and os.path.isdir(training_args.model_name_or_path)
        and any(
            os.path.isfile(os.path.join(training_args.model_name_or_path, f))
            for f in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
        )
    )

    if _has_weights:
        print(f"Loading model from: {training_args.model_name_or_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True,
            dtype=torch_dtype,
            attn_implementation=init_kwargs.get("attn_implementation", None),
        )
        print(f"[HY4] Model loaded successfully via from_pretrained.")
    else:
        if training_args.model_name_or_path is None:
            raise ValueError(
                "--model_name_or_path must be specified. Cannot load model config from None. "
                "Please provide the path to the model directory."
            )
        print(f"Model weights not found at: {training_args.model_name_or_path}, "
              f"using random initialized model instead.")
        config = transformers.AutoConfig.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True
        )
        model = transformers.AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            dtype=torch_dtype,
            attn_implementation=init_kwargs.get("attn_implementation", None),
        )
    
    if model_args.use_lora:
        # HY4 uses MLA (Multi-head Latent Attention) with different projection names
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # Fix: Mark PEFT LoRA wrapper modules as ZeRO-3 leaf modules.
        # PEFT wraps target Linear layers with lora.Linear, adding extra
        # sub-modules (base_layer, lora_A, lora_B). This changes the module
        # tree structure and disrupts ZeRO-3's parameter fetch/release
        # scheduling, causing OOM during backward recomputation.
        # By marking these wrappers as z3_leaf, ZeRO-3 treats them as atomic
        # units (same as the original Linear), restoring correct scheduling.
        from deepspeed.utils import set_z3_leaf_module
        from peft.tuners.lora import Linear as LoraLinear
        z3_leaf_count = 0
        for module in model.modules():
            if isinstance(module, LoraLinear):
                set_z3_leaf_module(module, True)
                z3_leaf_count += 1
        print(f"[z3_leaf] Marked {z3_leaf_count} LoraLinear modules with _z3_leaf=True", flush=True)

        # Verify the attribute is actually set
        verified_count = 0
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear):
                has_attr = getattr(module, '_z3_leaf', False)
                if has_attr:
                    verified_count += 1
                else:
                    print(f"[z3_leaf] WARNING: module '{name}' is LoraLinear but _z3_leaf={has_attr}", flush=True)
        print(f"[z3_leaf] Verification after marking: {verified_count}/{z3_leaf_count} modules have _z3_leaf=True", flush=True)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    training_args.lr_scheduler_kwargs = {
        'min_lr_rate': training_args.min_lr / training_args.learning_rate,
    }

    # -----------------------------------------------------------------------
    # Fix: DeepSpeed ZeRO-3 + gradient checkpointing compatibility.
    #
    # PyTorch's torch.utils.checkpoint with use_reentrant=False (the default
    # in transformers) performs strict metadata checks on recomputed tensors
    # during backward.  Under ZeRO-3, parameters are all-gathered during the
    # first forward pass (shape=[full_size]) but may be partitioned back
    # (shape=[0]) when the checkpoint recomputes, causing a CheckpointError.
    #
    # Setting use_reentrant=True avoids this strict metadata check.
    # -----------------------------------------------------------------------
    if training_args.gradient_checkpointing and training_args.deepspeed:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    trainer = Trainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args,
        callbacks=[CustomSaveCallback],
        **data_module
    )
    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # Monkey-patch: fix dtype mismatch in DeepSpeed ZeRO-3 linear wrapper.
    #
    # By this point the DeepSpeed engine has been initialised by the Trainer
    # and torch.nn.functional.linear has been replaced with
    # zero3_linear_wrap.  That wrapper does NOT auto-align input/weight
    # dtypes before the matmul, causing "expected mat1 and mat2 to have the
    # same dtype" errors in mixed-precision paths (e.g. enable_lm_head_fp32
    # casts input to fp32 but weight remains bf16 under ZeRO-3).
    #
    # We wrap F.linear HERE (after DeepSpeed init) so that:
    #   1. We are sure to capture the already-replaced function.
    #   2. The dtype cast happens *outside* the autograd.Function, so
    #      gradient-checkpointing recompute sees identical tensor metadata.
    # -----------------------------------------------------------------------
    import torch.nn.functional as _F
    _orig_F_linear = _F.linear

    def _dtype_safe_linear(input, weight, bias=None):
        if input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return _orig_F_linear(input, weight, bias)

    _F.linear = _dtype_safe_linear
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Monkey-patch: skip grad norm calculation when max_grad_norm == 0.
    #
    # Under ZeRO-3 + CPU offload, DeepSpeed's complete_grad_norm_calculation
    # all-gathers every gradient on CPU and does an ALLREDUCE to compute the
# global L2 norm. For a large model this is extremely slow and triggers an
    # NCCL ALLREDUCE timeout (NumelIn=1) at optimizer step, even when clipping
    # is disabled via gradient_clipping=0.0 (which only skips the clip, not the
    # computation).
    #
    # When the user explicitly sets --max_grad_norm 0 we fully skip the norm
    # computation (no all-gather, no ALLREDUCE) by returning 0.0 early.
    #
    # NOTE: We patch the CLASS method (not instance) because trainer.deepspeed
    # is None at this point — the DeepSpeed engine is created inside
    # trainer.train(). By patching the class, any future engine instance will
    # inherit the patched method.
    # -----------------------------------------------------------------------
    if getattr(training_args, "max_grad_norm", None) == 0:
        import torch as _torch
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as _ZeRO3Optimizer

        def _skip_get_norm_groups(self):
            return [_torch.tensor(0.0)]

        _ZeRO3Optimizer._get_norm_groups = _skip_get_norm_groups
        logging.info("[grad_norm] max_grad_norm=0: patched DeepSpeedZeroOptimizer_Stage3._get_norm_groups to skip norm computation")
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Post-DeepSpeed-init verification: check if _z3_leaf marks survived
    # DeepSpeed engine initialization (which happens inside Trainer.__init__).
    # -----------------------------------------------------------------------
    if model_args.use_lora:
        from peft.tuners.lora import Linear as LoraLinear
        post_init_count = 0
        post_init_verified = 0
        for name, module in trainer.model.named_modules():
            if isinstance(module, LoraLinear):
                post_init_count += 1
                has_attr = getattr(module, '_z3_leaf', False)
                if has_attr:
                    post_init_verified += 1
                elif post_init_count <= 5:  # Only print first few warnings to avoid spam
                    print(f"[z3_leaf] POST-INIT WARNING: module '{name}' lost _z3_leaf after Trainer init!", flush=True)
        print(f"[z3_leaf] Post-Trainer-init verification: {post_init_verified}/{post_init_count} LoraLinear modules still have _z3_leaf=True", flush=True)
    # -----------------------------------------------------------------------

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Synchronize all processes before exit to avoid "Connection reset by peer"
    # warnings caused by timing differences in multi-node shutdown.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    train()
