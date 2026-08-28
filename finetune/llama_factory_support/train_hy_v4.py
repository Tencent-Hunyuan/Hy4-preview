"""
LLaMA Factory training entry-point wrapper.

This script:
  1. Registers the chat template
  2. Applies all monkey-patches (checkpoint key rename, dtype fix, etc.)
  3. Injects HYV4PatchCallback into the training loop
  4. Calls run_exp() to start LLaMA Factory training

"""

import sys
import os

# Add current directory to path so patches can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 1: Register HYV4 template (must be before training starts)
import hy_v4_template  # noqa: F401

# Step 2: Apply checkpoint key rename patch (must be before model loading)
import hy_v4_patches  # noqa: F401

# Step 3: Inject HYV4PatchCallback into LLaMA Factory's training flow
from llamafactory.train.sft.workflow import run_sft as _orig_run_sft


def _patched_run_sft(
    model_args, data_args, training_args,
    finetuning_args, generating_args, callbacks=None
):
    """Wrap run_sft to inject HYV4PatchCallback."""
    if callbacks is None:
        callbacks = []

    # Determine tokenizer directory for the save callback
    tokenizer_dir = getattr(model_args, "model_name_or_path", None)
    callbacks.append(
        hy_v4_patches.HYV4PatchCallback(tokenizer_dir=tokenizer_dir)
    )

    return _orig_run_sft(
        model_args, data_args, training_args,
        finetuning_args, generating_args,
        callbacks=callbacks
    )


# Monkey-patch the SFT workflow
import llamafactory.train.sft.workflow as _sft_wf
_sft_wf.run_sft = _patched_run_sft


def _apply_skip_grad_norm_patch():
    """Skip grad norm computation for DeepSpeed ZeRO-3 + CPU offload.

    Under ZeRO-3 + CPU offload, DeepSpeed's complete_grad_norm_calculation
    all-gathers every gradient on CPU and does an ALLREDUCE to compute the
    global L2 norm. For a 770B model this is extremely slow and can cause
    NCCL timeout/deadlock at optimizer step.

    When max_grad_norm=0 (no clipping), we fully skip the norm computation
    by patching _get_norm_groups to return 0.0 immediately.
    """
    import torch
    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

        def _skip_get_norm_groups(self):
            return [torch.tensor(0.0)]

        DeepSpeedZeroOptimizer_Stage3._get_norm_groups = _skip_get_norm_groups
        print("[HYV4 Patch] Patched DeepSpeedZeroOptimizer_Stage3._get_norm_groups "
              "to skip grad norm computation (max_grad_norm=0).", flush=True)
    except ImportError:
        # DeepSpeed not available, skip
        pass


def main():
    """Entry point: called by torchrun in each worker process.

    Since train_lf.sh launches us via torchrun directly, all patches
    (template registration, checkpoint key rename, SFT callback injection)
    are already applied in this process.  We just call run_exp() to start
    training — no need to go through the CLI launcher.
    """
    # Apply grad norm skip patch for DeepSpeed ZeRO-3 full SFT.
    # This must be done before Trainer creates the DeepSpeed engine.
    # The patch is safe even when not using DeepSpeed (it's a no-op if
    # DeepSpeed is not imported or ZeRO-3 is not used).
    _apply_skip_grad_norm_patch()

    from llamafactory.train.tuner import run_exp
    run_exp()


if __name__ == "__main__":
    main()
