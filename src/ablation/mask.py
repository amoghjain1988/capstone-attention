"""Turn a list of removed edges into a mask, and run a masked prediction.

See CONTRACT.md section 5.7 for the signatures.

Entry (dst, src) of the mask is the edge FROM src INTO dst. The row index is
the agent that attends. The column index is the agent it attends to. This
matches attn[dst, src] and matches the src and dst columns of the
attention_edges table. See src/models/base.py, check_edge_mask.

Do not expand the mask to an 8 by 8 time block. AgentFormer already tiles the
(N, N) matrix internally. See docs/EDA_FINDINGS.md fact F15.
"""

from __future__ import annotations

import numpy as np

import config
from src.models import base


def build_mask(n_agents: int, remove: list[int], ego_index: int) -> np.ndarray:
    """Return an (N, N) boolean mask. True keeps the edge.

    `remove` holds agent indices on the window axis, not agent ids. Each
    index names a neighbour whose edge into the ego is set False. Only
    entries whose dst is `ego_index` are ever set False. The diagonal always
    stays True.
    """
    if n_agents < 1:
        raise ValueError(f"n_agents must be positive, it is {n_agents}")
    if not (0 <= ego_index < n_agents):
        raise ValueError(
            f"ego_index must be inside 0..{n_agents - 1}, it is {ego_index}"
        )

    mask = np.ones((n_agents, n_agents), dtype=bool)

    for raw_src in remove:
        src = int(raw_src)
        if not (0 <= src < n_agents):
            raise ValueError(
                f"remove holds an index outside 0..{n_agents - 1}: {src}"
            )
        if src == ego_index:
            raise ValueError("remove must not name the ego. The ego has no self edge.")
        mask[ego_index, src] = False

    np.fill_diagonal(mask, True)

    # Only the ego row may hold a False entry. Assert it, per CONTRACT.md.
    cleared_rows = np.flatnonzero(~mask.all(axis=1))
    assert set(cleared_rows.tolist()) <= {ego_index}, (
        "build_mask cleared an entry outside the ego row"
    )

    return mask


def masked_predict(
    model: base.MaskablePredictor,
    hist: np.ndarray,
    edge_mask: np.ndarray,
    draws: np.ndarray,
    mask_policy: str,
) -> np.ndarray:
    """Return (K, N, 12, 2). Run one prediction under one mask.

    Pass `edge_mask` and `draws` straight through to `model.predict`. Use the
    SAME `draws` for every arm of the same window, so common random numbers
    make the shifts comparable. Raise `ValueError` when `mask_policy` is not
    one of `config.MASK_POLICIES`.

    This function also passes `mask_policy` to `model.predict`, so a model
    that honours the two policies differently gives two different answers.
    Before this fix, `mask_policy` reached this function and stopped here, so
    `weight_zero` silently ran whatever the model's one behaviour was. That
    is a silent duplicate, and it is wrong.

    A model whose `predict` method does not accept `mask_policy` at all
    cannot tell the two policies apart. `MockPredictor` is one such model
    today. In that one case, and only for `config.PRIMARY_MASK_POLICY`, this
    function calls `model.predict` without the argument, since that is the
    one behaviour such a model already has. Any OTHER policy against a model
    like that raises `NotImplementedError`. It never again returns a silent
    duplicate of the primary policy under a different name.

    A model that ACCEPTS `mask_policy` and then ignores its value, the way
    this function used to, is not caught here. See
    `tests/test_mask.py::test_a_model_that_drops_mask_policy_is_not_caught_by_this_guard`
    for that limit, stated plainly rather than assumed away.
    """
    if mask_policy not in config.MASK_POLICIES:
        raise ValueError(
            f"mask_policy must be one of {config.MASK_POLICIES}, it is {mask_policy!r}"
        )

    try:
        return model.predict(
            hist, edge_mask=edge_mask, draws=draws, mask_policy=mask_policy
        )
    except TypeError as error:
        if "mask_policy" not in str(error):
            raise
        if mask_policy != config.PRIMARY_MASK_POLICY:
            raise NotImplementedError(
                f"{type(model).__name__}.predict does not accept mask_policy, so "
                f"it cannot honour {mask_policy!r} apart from its one behaviour. "
                f"Only {config.PRIMARY_MASK_POLICY!r} may run against a model "
                "like this."
            ) from error
        return model.predict(hist, edge_mask=edge_mask, draws=draws)
