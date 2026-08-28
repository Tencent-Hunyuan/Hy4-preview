"""CPU regression tests for the ZeRO-3 shard-by-shard checkpoint loader in
hy_v4_patches.py.

Covers:
- exact expert-identity preservation during per-expert -> 3D fusion
  (sentinel values per expert, not just tensor shapes);
- fail-closed behavior for missing experts / missing projections;
- exact, index-derived expert-group completeness (no premature fusion when
  an expert group spans several shards);
- propagation of the (error_msgs, missing_keys) contract of
  transformers.integrations.deepspeed._load_state_dict_into_zero3_model
  (which reports e.g. size mismatches instead of raising);
- keys only being counted as loaded after the load actually succeeded.

Run:  pytest finetune/llama_factory_support/tests -q
"""
import os

import pytest
import torch

import conftest as H
from conftest import (
    EXPERT_PREFIX,
    HIDDEN,
    MOE_INTER,
    NUM_EXPERTS,
    fake_zero3_enabled,
)


def _group(mod, keys=None, values=None):
    """Build {expert_idx -> {proj -> tensor}} from sentinel per-expert keys."""
    per_expert, _, _ = H.expert_sentinels()
    if keys is None:
        keys = list(per_expert)
    group = {}
    for k in keys:
        m = mod._EXPERT_KEY_RE.match(k)
        group.setdefault(int(m.group(2)), {})[m.group(3)] = per_expert[k]
    return group


# ---------------------------------------------------------------------------
# Unit: _fuse_expert_group / _validate_expert_group
# ---------------------------------------------------------------------------

class TestFusion:
    def test_complete_group_identity_and_shape(self, hy_patches):
        gate_up, down = hy_patches._fuse_expert_group(EXPERT_PREFIX, _group(hy_patches))
        # Packed layout: gate_up [E, 2I, H] with gate rows before up rows;
        # down [E, H, I]. Matches the official checkpoint tensors
        # (e.g. [256, 4096, 6144] / [256, 6144, 2048]).
        assert gate_up.shape == (NUM_EXPERTS, 2 * MOE_INTER, HIDDEN)
        assert down.shape == (NUM_EXPERTS, HIDDEN, MOE_INTER)
        for i in range(NUM_EXPERTS):
            assert torch.all(gate_up[i, :MOE_INTER] == float(i)), "gate row identity"
            assert torch.all(gate_up[i, MOE_INTER:] == i + 0.5), "up row identity"
            assert torch.all(down[i] == i + 0.25), "down row identity"

    def test_missing_middle_expert_raises(self, hy_patches):
        group = _group(hy_patches, H.expert_keys(skip=set(H.expert_keys([1]))))
        with pytest.raises(RuntimeError, match=r"missing experts: \[1\]"):
            hy_patches._fuse_expert_group(
                EXPERT_PREFIX, group, expected_ids=range(NUM_EXPERTS)
            )

    def test_missing_final_expert_raises_with_expected_ids(self, hy_patches):
        # A truncated 0..k prefix is undetectable from the tensors alone;
        # the index-derived expected_ids catch it.
        group = _group(hy_patches, H.expert_keys(range(NUM_EXPERTS - 1)))
        with pytest.raises(RuntimeError, match=r"missing experts: \[3\]"):
            hy_patches._fuse_expert_group(
                EXPERT_PREFIX, group, expected_ids=range(NUM_EXPERTS)
            )

    @pytest.mark.parametrize("proj", ["gate_proj", "up_proj", "down_proj"])
    def test_missing_projection_raises(self, hy_patches, proj):
        # The pre-fix code appended gate_up and down rows under independent
        # conditions, so a single missing projection desynchronized
        # gate_up_proj[i] from down_proj[i]. It must be a hard error.
        skip = {f"{EXPERT_PREFIX}1.{proj}.weight"}
        group = _group(hy_patches, H.expert_keys(skip=skip))
        with pytest.raises(RuntimeError, match=rf"expert 1 missing projections: \['{proj}'\]"):
            hy_patches._fuse_expert_group(EXPERT_PREFIX, group)

    def test_non_contiguous_ids_raise_without_expected(self, hy_patches):
        group = _group(hy_patches, H.expert_keys([0, 1, 3]))
        with pytest.raises(RuntimeError, match=r"missing experts: \[2\]"):
            hy_patches._fuse_expert_group(EXPERT_PREFIX, group)

    def test_unexpected_extra_expert_raises(self, hy_patches):
        group = _group(hy_patches)
        with pytest.raises(RuntimeError, match=r"unexpected expert ids: \[3\]"):
            hy_patches._fuse_expert_group(
                EXPERT_PREFIX, group, expected_ids=range(NUM_EXPERTS - 1)
            )


# ---------------------------------------------------------------------------
# Unit: _derive_expected_expert_groups / _check_zero3_load_result
# ---------------------------------------------------------------------------

class TestContracts:
    def test_derive_expected_groups_inner_format(self, hy_patches):
        wm = {k: "s1" for k in H.expert_keys()}
        wm["model.norm.weight"] = "s2"
        expected = hy_patches._derive_expected_expert_groups(wm)
        assert set(expected) == {EXPERT_PREFIX}
        assert sorted(expected[EXPERT_PREFIX]) == list(range(NUM_EXPERTS))
        assert all(
            projs == set(H.PROJS) for projs in expected[EXPERT_PREFIX].values()
        )

    def test_derive_expected_groups_outer_format_is_empty(self, hy_patches):
        # The official Hy4 checkpoint stores pre-fused tensors; the state
        # machine must stay inert for it.
        wm = {
            f"{EXPERT_PREFIX}gate_up_proj": "s1",
            f"{EXPERT_PREFIX}down_proj": "s1",
            "model.norm.weight": "s1",
        }
        assert hy_patches._derive_expected_expert_groups(wm) == {}

    def test_zero3_error_msgs_raise(self, hy_patches):
        # Contract of transformers >= 5.x: (error_msgs, missing_keys).
        result = (["size mismatch for model.norm.weight: ..."], set())
        with pytest.raises(RuntimeError, match="size mismatch"):
            hy_patches._check_zero3_load_result(result, "shard x")

    def test_zero3_clean_result_passes(self, hy_patches):
        # Per-shard missing_keys are expected (each shard covers a subset of
        # the model) and must NOT be treated as an error.
        hy_patches._check_zero3_load_result(([], {"model.some.other.key"}), "shard x")

    def test_zero3_legacy_list_contract(self, hy_patches):
        with pytest.raises(RuntimeError, match="boom"):
            hy_patches._check_zero3_load_result(["boom"], "shard x")
        hy_patches._check_zero3_load_result([], "shard x")


# ---------------------------------------------------------------------------
# Unit: failed keys must not be counted as loaded
# ---------------------------------------------------------------------------

class TestLoadAccounting:
    def test_failed_load_is_not_marked_loaded(self, hy_patches):
        # The pre-fix loader did all_loaded_keys.update(...) unconditionally,
        # so a failed shard was still reported as loaded. The fixed loader
        # checks the result first; simulate the sequence here.
        all_loaded_keys = set()
        renamed_sd = {"model.norm.weight": torch.zeros(16)}
        result = (["size mismatch for model.norm.weight"], set())
        with pytest.raises(RuntimeError):
            hy_patches._check_zero3_load_result(result, "shard bad")
            all_loaded_keys.update(renamed_sd.keys())  # unreachable, as in the fix
        assert all_loaded_keys == set()


# ---------------------------------------------------------------------------
# Integration: full shard-by-shard loads through the patched from_pretrained
# (real transformers ZeRO-3 loader; see conftest for the stubbed pieces)
# ---------------------------------------------------------------------------

def _assert_expert_weights_exact(model, expected_sd):
    got = model.state_dict()
    for part in ("gate_up_proj", "down_proj"):
        key = f"{EXPERT_PREFIX}{part}"
        assert torch.equal(got[key].float(), expected_sd[key]), key


class TestShardLoading:
    def test_complete_set_single_shard(self, hy_patches, tmp_path):
        keys = H.expert_keys()
        expected = H.build_inner_checkpoint(tmp_path / "m", [keys])
        model = H.run_loader(tmp_path / "m")
        _assert_expert_weights_exact(model, expected)
        # Buffers (e_score_correction_bias) must load too.
        got = model.state_dict()
        assert torch.equal(
            got["model.layers.1.mlp.e_score_correction_bias"].float(),
            expected["model.layers.1.mlp.e_score_correction_bias"],
        )
        assert torch.equal(
            got["model.norm.weight"].float(), expected["model.norm.weight"]
        )

    def test_boundary_exactly_between_experts(self, hy_patches, tmp_path):
        # Experts 0-1 in shard 1, experts 2-3 in shard 2. The pre-fix
        # heuristic (len == max + 1, all projections present) fused after
        # shard 1 with only 2 of 4 experts; the resulting size mismatch was
        # silently discarded and the expert weights were never loaded.
        keys = H.expert_keys()
        expected = H.build_inner_checkpoint(tmp_path / "m", [keys[:6], keys[6:]])
        model = H.run_loader(tmp_path / "m")
        _assert_expert_weights_exact(model, expected)

    def test_boundary_inside_one_experts_projections(self, hy_patches, tmp_path):
        # Expert 1's projections are split across two shards.
        keys = H.expert_keys()
        assert keys[4].startswith(f"{EXPERT_PREFIX}1.")
        expected = H.build_inner_checkpoint(tmp_path / "m", [keys[:4], keys[4:]])
        model = H.run_loader(tmp_path / "m")
        _assert_expert_weights_exact(model, expected)

    def test_shuffled_shard_order(self, hy_patches, tmp_path):
        # Shards arrive in an order where high expert ids come first --
        # mirrors the official index, whose weight_map order is not sorted.
        keys = H.expert_keys()
        batches = [keys[9:], keys[3:6], keys[:3], keys[6:9]]
        expected = H.build_inner_checkpoint(tmp_path / "m", batches)
        model = H.run_loader(tmp_path / "m")
        _assert_expert_weights_exact(model, expected)

    def test_one_key_per_shard(self, hy_patches, tmp_path):
        keys = H.expert_keys()
        expected = H.build_inner_checkpoint(tmp_path / "m", [[k] for k in keys])
        model = H.run_loader(tmp_path / "m")
        _assert_expert_weights_exact(model, expected)


class TestFailClosed:
    def test_partial_contiguous_prefix_fails(self, hy_patches, tmp_path):
        # Checkpoint only contains experts 0..2 of 4: exactly the case the
        # old contiguity heuristic accepted as complete.
        H.build_inner_checkpoint(
            tmp_path / "m", [H.expert_keys(range(NUM_EXPERTS - 1))]
        )
        with pytest.raises(RuntimeError, match="expert"):
            H.run_loader(tmp_path / "m")

    def test_missing_middle_expert_fails(self, hy_patches, tmp_path):
        keys = H.expert_keys(skip=set(H.expert_keys([2])))
        H.build_inner_checkpoint(tmp_path / "m", [keys])
        with pytest.raises(RuntimeError, match="expert"):
            H.run_loader(tmp_path / "m")

    @pytest.mark.parametrize("proj", ["gate_proj", "up_proj", "down_proj"])
    def test_missing_projection_fails(self, hy_patches, tmp_path, proj):
        keys = H.expert_keys(skip={f"{EXPERT_PREFIX}2.{proj}.weight"})
        H.build_inner_checkpoint(tmp_path / "m", [keys])
        with pytest.raises(RuntimeError):
            H.run_loader(tmp_path / "m")

    def test_underlying_loader_shape_error_fails(self, hy_patches, tmp_path):
        # A wrong-shaped non-expert tensor: the real transformers ZeRO-3
        # loader reports it via error_msgs; the loader must raise instead of
        # discarding it (pre-fix behavior: silent, key counted as loaded).
        sd, _ = H.reference_state_dict()
        sd["model.norm.weight"] = torch.zeros(16)  # model expects (8,)
        H.write_sharded_checkpoint(tmp_path / "m", [sd])
        with pytest.raises(RuntimeError, match="size mismatch"):
            H.run_loader(tmp_path / "m")

    def test_index_config_expert_count_mismatch_fails(self, hy_patches, tmp_path):
        # config says 4 experts; index/checkpoint provides 3 (0..2). Caught
        # up front by cross-validation, before reading any tensor data.
        H.build_inner_checkpoint(
            tmp_path / "m", [H.expert_keys(range(3))]
        )
        with pytest.raises(
            RuntimeError, match="provides 3 experts but model config expects 4"
        ):
            H.run_loader(tmp_path / "m")


# ---------------------------------------------------------------------------
# Integration: Patch 1 (_load_state_dict_into_zero3_model wrapper) fusion
# ---------------------------------------------------------------------------

class TestPatch1Fusion:
    def _run_patched_zero3(self, sd):
        import transformers.integrations.deepspeed as hf_ds
        from transformers import AutoModelForCausalLM

        H._init_single_process_gloo()
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(
            H.tiny_config(), dtype=torch.float32
        )
        with fake_zero3_enabled():
            result = hf_ds._load_state_dict_into_zero3_model(model, sd)
        return model, result

    def test_per_expert_state_dict_fused_and_loaded(self, hy_patches):
        expected, per_expert = H.reference_state_dict()
        sd = {
            k: v for k, v in expected.items()
            if not k.startswith(EXPERT_PREFIX)
        }
        sd.update(per_expert)
        model, result = self._run_patched_zero3(sd)
        hy_patches._check_zero3_load_result(result, "test")
        _assert_expert_weights_exact(model, expected)

    def test_missing_expert_raises(self, hy_patches):
        expected, per_expert = H.reference_state_dict()
        sd = {
            k: v for k, v in per_expert.items()
            if not k.startswith(f"{EXPERT_PREFIX}1.")
        }
        with pytest.raises(RuntimeError, match=r"missing experts: \[1\]"):
            self._run_patched_zero3(sd)

    def test_missing_down_proj_raises_instead_of_desync(self, hy_patches):
        # Pre-fix: gate_up got 4 rows, down got 3 -> down_proj[1] silently
        # became expert 2's weights (and the shape error was left to the
        # caller, or swallowed entirely in the Patch 3 path).
        _, per_expert = H.reference_state_dict()
        sd = dict(per_expert)
        del sd[f"{EXPERT_PREFIX}1.down_proj.weight"]
        with pytest.raises(
            RuntimeError, match=r"expert 1 missing projections: \['down_proj'\]"
        ):
            self._run_patched_zero3(sd)

# NOTE: a duplicate projection for the same expert is not reachable through a
# safetensors checkpoint (the index's weight_map and each shard's key set are
# maps, so a key exists at most once; duplicated across shards it would be the
# same key, which safetensors forbids within one file and the index cannot
# express for two files).


# ---------------------------------------------------------------------------
# Distributed invariant: the loader's error channel is rank-local (only the
# rank performing the copy observes error_msgs; verified on a real 2-rank
# ZeRO-3 run), so _check_zero3_load_result must make EVERY rank raise when
# ANY rank saw errors. Verified here with a real 2-process gloo group.
# ---------------------------------------------------------------------------

def _dist_sync_worker(rank, world_size, port, patches_path, out_dir):
    """Subprocess worker: rank 0 feeds an error result, rank 1 a clean one;
    both must raise."""
    import importlib.util
    import os

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    spec = importlib.util.spec_from_file_location("hyp_dist_child", patches_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if rank == 0:
        result = (["size mismatch for model.norm.weight: ..."], set())
    else:
        result = ([], set())

    outcome = "no-raise"
    try:
        mod._check_zero3_load_result(result, "shard test")
    except RuntimeError as e:
        outcome = "raised-local" if "size mismatch" in str(e) else "raised-remote"
    with open(os.path.join(out_dir, f"rank{rank}.txt"), "w") as f:
        f.write(outcome)
    dist.barrier()
    dist.destroy_process_group()


class TestDistributedErrorSync:
    def test_all_ranks_raise_when_one_rank_sees_errors(self, tmp_path):
        import socket

        import torch.multiprocessing as mp

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        patches_path = os.environ.get("HYV4_PATCHES_PATH", H.DEFAULT_PATCHES_PATH)
        mp.spawn(
            _dist_sync_worker,
            args=(2, port, patches_path, str(tmp_path)),
            nprocs=2,
            join=True,
        )
        outcomes = {
            r: (tmp_path / f"rank{r}.txt").read_text() for r in range(2)
        }
        assert outcomes[0] == "raised-local", outcomes
        assert outcomes[1] == "raised-remote", outcomes

    def test_single_process_no_collective_needed(self, hy_patches):
        # torch.distributed initialized with world_size == 1 (as in this CPU
        # suite) or not at all: the check must work without any collective.
        with pytest.raises(RuntimeError, match="size mismatch"):
            hy_patches._check_zero3_load_result((["size mismatch for x"], set()), "s")
        hy_patches._check_zero3_load_result(([], set()), "s")
