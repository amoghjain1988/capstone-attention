"""The two mask policies give two real answers on the real model.

See CONTRACT.md section 5.7 and docs/FINISH_PLAN.md section 3.12.

logit_neg_inf sets the logit of a removed edge to minus infinity, so the
softmax gives that edge a weight of exactly 0 and the row still sums to 1.
weight_zero leaves every logit alone and multiplies the weight by 0 after the
softmax, so the row loses the removed mass. The two rules therefore answer two
different questions, and this file proves that the code really tells them
apart. Before the weight_zero policy existed, a robustness check read two
identical numbers as strong evidence. That is the bug this file guards.

Every test here needs a CUDA device, the AgentFormer checkpoints and the
windows table. Each one is absent on a plain CPU machine, so the module skips
itself and reports no failure.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from src.ablation.mask import build_mask
from src.ablation.shift import ego_path, shift
from src.data import schema

SCENE = "zara1"
MIN_AGENTS = 5

GPU_READY = (config.CHECKPOINT_DIR / "manifest.json").exists()
WINDOWS_READY = schema.path_of("windows", config.PROCESSED_DIR).exists()

pytestmark = [
    pytest.mark.skipif(not GPU_READY, reason="run scripts/setup_agentformer.py"),
    pytest.mark.skipif(not WINDOWS_READY, reason="build the windows table first"),
]


# ---------------------------------------------------------------------------
# Fixtures. One model and one forward pass serve the whole module.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_module():
    """Return torch, or skip when no CUDA device is present."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    return torch


@pytest.fixture(scope="module")
def window():
    """Return the first eligible zara1 window that holds at least 5 agents."""
    table = schema.read("windows", config.PROCESSED_DIR)
    rows = table[
        (table["scene"].astype(str) == SCENE)
        & table["eligible"].astype(bool)
        & (table["n_agents"].astype(int) >= MIN_AGENTS)
    ]
    if rows.empty:
        pytest.skip(f"no eligible {SCENE} window holds {MIN_AGENTS} agents")
    return rows.sort_values("window_id").iloc[0]


@pytest.fixture(scope="module")
def model(torch_module):
    """Return the zara1 checkpoint on the GPU."""
    from src.models.agentformer import AgentFormerPredictor

    return AgentFormerPredictor(SCENE, device="cuda")


@pytest.fixture(scope="module")
def setup(window):
    """Return the history, the ego position and the mask of one removed edge.

    The removed edge is the one from the nearest neighbour at the last
    observed frame into the ego. The nearest neighbour carries the strongest
    interaction, so both policies must move the ego by more than 0.
    """
    hist = schema.unpack_hist(window, config.N_HIST)
    ego_pos = schema.ego_index(window)
    n_agents = int(hist.shape[0])

    distance = np.linalg.norm(hist[:, -1, :] - hist[ego_pos, -1, :], axis=-1)
    distance[ego_pos] = np.inf
    neighbour = int(np.argmin(distance))

    mask = build_mask(n_agents, [neighbour], ego_pos)
    return {
        "window_id": str(window["window_id"]),
        "hist": hist,
        "ego_pos": ego_pos,
        "n_agents": n_agents,
        "neighbour": neighbour,
        "mask": mask,
    }


@pytest.fixture(scope="module")
def shifts(model, setup):
    """Return the ego shift of each policy on the same removed edge."""
    hist, ego_pos = setup["hist"], setup["ego_pos"]
    unmasked = ego_path(model.predict(hist), ego_pos)

    out = {}
    for policy in config.MASK_POLICIES:
        masked = model.predict(hist, edge_mask=setup["mask"], mask_policy=policy)
        out[policy] = shift(ego_path(masked, ego_pos), unmasked)

    print(
        f"\nwindow {setup['window_id']}, N={setup['n_agents']}, "
        f"ego index {ego_pos}, removed src index {setup['neighbour']}"
    )
    for policy, value in out.items():
        print(f"  {policy:<14} shift = {value:.6f} m")
    return out


# ---------------------------------------------------------------------------
# (a) An all-True mask must change nothing, under either policy
# ---------------------------------------------------------------------------


def test_an_all_true_mask_reproduces_the_unmasked_prediction(model, setup):
    """A mask that removes no edge must give back the unmasked path exactly.

    weight_zero multiplies every weight by 1.0, and that product is exact in
    binary arithmetic, so the two arrays must match bit for bit.
    """
    hist = setup["hist"]
    keep_all = np.ones((setup["n_agents"], setup["n_agents"]), dtype=bool)
    unmasked = model.predict(hist)

    for policy in config.MASK_POLICIES:
        kept = model.predict(hist, edge_mask=keep_all, mask_policy=policy)
        assert np.array_equal(kept, unmasked), policy


def test_an_all_true_mask_gives_a_shift_of_exactly_zero(model, setup):
    """The same statement in the unit the study reports: metres."""
    hist, ego_pos = setup["hist"], setup["ego_pos"]
    keep_all = np.ones((setup["n_agents"], setup["n_agents"]), dtype=bool)
    unmasked = ego_path(model.predict(hist), ego_pos)

    for policy in config.MASK_POLICIES:
        kept = model.predict(hist, edge_mask=keep_all, mask_policy=policy)
        assert shift(ego_path(kept, ego_pos), unmasked) == 0.0, policy


# ---------------------------------------------------------------------------
# (b) The two policies must give two different shifts, both above 0
# ---------------------------------------------------------------------------


def test_both_policies_move_the_ego(shifts):
    """A removed edge must change the forecast under either rule.

    The released model is deterministic, so the noise floor is exactly 0 and
    any shift above 0 is a real change. See docs/EDA_FINDINGS.md finding F14.
    """
    for policy, value in shifts.items():
        assert value > 0.0, policy


def test_the_two_policies_give_two_different_shifts(shifts):
    """The regression test for the silent duplicate.

    If weight_zero ever falls back to the logit mask again, the two shifts
    become the same number and this assertion fails.
    """
    neg_inf = shifts["logit_neg_inf"]
    weight_zero = shifts["weight_zero"]
    assert neg_inf != pytest.approx(weight_zero, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# (c) The patched pass must repeat itself
# ---------------------------------------------------------------------------


def test_weight_zero_is_deterministic(model, setup):
    """Two identical weight_zero calls must give one identical array."""
    hist = setup["hist"]
    first = model.predict(hist, edge_mask=setup["mask"], mask_policy="weight_zero")
    second = model.predict(hist, edge_mask=setup["mask"], mask_policy="weight_zero")
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# (d) The patch of the third party module must be undone
# ---------------------------------------------------------------------------


def test_the_softmax_patch_is_undone(model, setup, torch_module):
    """After the call, agentformer_lib holds the real softmax again.

    The wrapper lives only for the length of one predict() call. A leak would
    silently mask every later forward pass. The unmasked reference is one such
    pass, so this test is the guard on the finally block.
    """
    from src.models import agentformer as module

    model.predict(hist=setup["hist"], edge_mask=setup["mask"], mask_policy="weight_zero")
    with module._inside_agentformer():
        library = module.agentformer_lib()
    assert library.softmax is torch_module.nn.functional.softmax


def test_the_patch_is_undone_after_a_failed_call(model, setup, torch_module):
    """An error inside the forward pass must still restore the softmax."""
    from src.models import agentformer as module

    broken = np.zeros((setup["n_agents"], setup["n_agents"] + 1), dtype=bool)
    with pytest.raises(ValueError, match="edge_mask must be"):
        model.predict(setup["hist"], edge_mask=broken, mask_policy="weight_zero")

    with module._inside_agentformer():
        library = module.agentformer_lib()
    assert library.softmax is torch_module.nn.functional.softmax


def test_an_unknown_policy_raises(model, setup):
    """Only the two names of config.MASK_POLICIES may reach the model."""
    with pytest.raises(NotImplementedError, match="not_a_policy"):
        model.predict(
            setup["hist"], edge_mask=setup["mask"], mask_policy="not_a_policy"
        )
