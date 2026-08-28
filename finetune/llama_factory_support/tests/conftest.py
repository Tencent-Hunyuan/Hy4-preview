"""CPU-only test harness for hy_v4_patches.py.

No GPU, no Hy4 weights, no DeepSpeed install and no distributed cluster are
required:

- A stub ``deepspeed`` module is installed in sys.modules (no-op zero.Init /
  GatheredParameters). Under single-process CPU testing ZeRO-3 partitioning
  is a no-op anyway; everything else (transformers'
  _load_state_dict_into_zero3_model, its error_msgs channel, the hy_v4
  shard loader) runs for real.
- torch.distributed is initialized as a single-process gloo group so the
  real ZeRO-3 loader's rank-0 copy path executes.
- transformers.integrations.deepspeed.is_deepspeed_zero3_enabled /
  deepspeed_config are overridden so the shard-by-shard code path engages.

The tiny model uses the hy_v3 architecture (native to the supported
transformers), which shares Hy4's fused expert layout:
    experts.gate_up_proj: [num_experts, 2 * moe_intermediate, hidden]
    experts.down_proj:    [num_experts, hidden, moe_intermediate]

Set HYV4_PATCHES_PATH to test a different copy of hy_v4_patches.py (e.g. for
bisecting against an older revision).
"""
import contextlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import types

import pytest
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATCHES_PATH = os.path.join(TESTS_DIR, os.pardir, "hy_v4_patches.py")

NUM_EXPERTS = 4
HIDDEN = 8
MOE_INTER = 4
EXPERT_PREFIX = "model.layers.1.mlp.experts."
PROJS = ("gate_proj", "up_proj", "down_proj")


def _install_deepspeed_stub():
    if "deepspeed" in sys.modules:
        return
    ds = types.ModuleType("deepspeed")
    zero = types.ModuleType("deepspeed.zero")

    class _NoOpCtx:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    zero.Init = _NoOpCtx
    zero.GatheredParameters = _NoOpCtx
    ds.zero = zero
    # A real ModuleSpec so importlib.util.find_spec("deepspeed") works.
    ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
    zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", loader=None)
    ds.__version__ = "0.18.7-stub"
    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.zero"] = zero


def _init_single_process_gloo():
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)


@pytest.fixture(scope="session")
def hy_patches():
    """Import the hy_v4_patches module under test (applies its monkey
    patches on import)."""
    _install_deepspeed_stub()
    path = os.environ.get("HYV4_PATCHES_PATH", DEFAULT_PATCHES_PATH)
    spec = importlib.util.spec_from_file_location("hy_v4_patches_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hy_v4_patches_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def fake_zero3_enabled():
    import transformers.integrations.deepspeed as hf_ds

    saved = (hf_ds.is_deepspeed_zero3_enabled, hf_ds.deepspeed_config)
    hf_ds.is_deepspeed_zero3_enabled = lambda: True
    hf_ds.deepspeed_config = lambda: {"zero_optimization": {"stage": 3}}
    try:
        yield
    finally:
        hf_ds.is_deepspeed_zero3_enabled, hf_ds.deepspeed_config = saved


def tiny_config(num_experts=NUM_EXPERTS):
    from transformers import HYV3Config

    return HYV3Config(
        hidden_size=HIDDEN,
        intermediate_size=16,
        moe_intermediate_size=MOE_INTER,
        num_experts=num_experts,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        mlp_layer_types=["dense", "sparse"],
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=64,
    )


def expert_sentinels():
    """Per-expert tensors where every value identifies its expert and
    projection: gate = i, up = i + 0.5, down = i + 0.25 (exact in bf16/fp32).

    Returns (per_expert_keys dict, fused gate_up [E, 2I, H], fused down
    [E, H, I])."""
    per_expert = {}
    gate_up = torch.empty(NUM_EXPERTS, 2 * MOE_INTER, HIDDEN)
    down = torch.empty(NUM_EXPERTS, HIDDEN, MOE_INTER)
    for i in range(NUM_EXPERTS):
        g = torch.full((MOE_INTER, HIDDEN), float(i))
        u = torch.full((MOE_INTER, HIDDEN), i + 0.5)
        d = torch.full((HIDDEN, MOE_INTER), i + 0.25)
        per_expert[f"{EXPERT_PREFIX}{i}.gate_proj.weight"] = g
        per_expert[f"{EXPERT_PREFIX}{i}.up_proj.weight"] = u
        per_expert[f"{EXPERT_PREFIX}{i}.down_proj.weight"] = d
        gate_up[i] = torch.cat([g, u], dim=0)
        down[i] = d
    return per_expert, gate_up, down


def reference_state_dict(seed=0):
    """Deterministic full reference state dict for the tiny model with expert
    sentinels. Returns (fused sd as the model stores it, per-expert keys)."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(tiny_config(), dtype=torch.float32)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    per_expert, gate_up, down = expert_sentinels()
    sd[f"{EXPERT_PREFIX}gate_up_proj"] = gate_up
    sd[f"{EXPERT_PREFIX}down_proj"] = down
    return sd, per_expert


def write_sharded_checkpoint(model_dir, shards, config=None):
    """Write safetensors shards + index + config.json. The weight_map's
    insertion order follows ``shards`` order, which is exactly what drives
    the loader's shard iteration order."""
    from safetensors.torch import save_file

    os.makedirs(model_dir, exist_ok=True)
    (config or tiny_config()).save_pretrained(model_dir)
    n = len(shards)
    weight_map = {}
    for si, shard in enumerate(shards, 1):
        fname = f"model-{si:05d}-of-{n:05d}.safetensors"
        save_file(
            {k: v.contiguous() for k, v in shard.items()},
            os.path.join(model_dir, fname),
        )
        for k in shard:
            weight_map[k] = fname
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return str(model_dir)


def build_inner_checkpoint(model_dir, expert_key_batches, config=None, seed=0):
    """Inner-format (per-expert) sharded checkpoint for the tiny model.

    expert_key_batches: list of per-expert key-name lists; batch i -> shard i.
    Non-expert keys go into the last shard. Returns the expected fused sd."""
    sd, per_expert = reference_state_dict(seed)
    non_expert = {k: v for k, v in sd.items() if not k.startswith(EXPERT_PREFIX)}
    shards = [{k: per_expert[k] for k in batch} for batch in expert_key_batches]
    shards[-1].update(non_expert)
    write_sharded_checkpoint(model_dir, shards, config=config)
    return sd


def run_loader(model_dir, dtype=torch.float32):
    """Invoke the patched AutoModelForCausalLM.from_pretrained end to end."""
    import transformers

    _init_single_process_gloo()
    with fake_zero3_enabled():
        # str() matters: the shard loader only engages for str paths.
        return transformers.AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=dtype
        )


def expert_keys(expert_ids=range(NUM_EXPERTS), projs=PROJS, skip=()):
    keys = [
        f"{EXPERT_PREFIX}{i}.{p}.weight" for i in expert_ids for p in projs
    ]
    return [k for k in keys if k not in skip]
