"""The real model. THE SWAP POINT.

See CONTRACT.md section 5.5 for the signatures.

Nobody imports AgentFormer outside this file.

THREE FACTS THAT CONTRADICT THE CONTRACT. All three are measured, not guessed.
See docs/EDA_FINDINGS.md findings F13, F14 and F15.

1. The token order is TIME-major, not agent-major.
   AgentFormer flattens (T, N, 2) with a plain view, so a token sits at
   time_index * N + agent_index. CONTRACT.md section 4.1 promises
   agent_index * T + time_index. attention() transposes both axes before it
   returns, so every caller still sees the agent-major order of the contract.

2. The released model is DETERMINISTIC at inference.
   DLow.inference() calls main(mean=True), which sets z = b and never draws an
   epsilon. Two identical calls give identical paths. CONTRACT.md section 4.2
   says the opposite. The `draws` argument therefore selects nothing, and the
   noise floor of src/faithfulness/sanity.py is exactly 0.

3. Only the encoder map is (N*8, N*8).
   The decoder self-attention is (N*12, N*12) and the cross attention is
   (N*12, N*8). PRIMARY_MODULE is the encoder, so the primary path matches the
   contract. Read the shape from the array, never assume it.

MASKING. CONTRACT.md section 5.7 names two policies. This file honours both.

logit_neg_inf. AgentFormer already carries an (N, N) additive float matrix in
data['agent_mask']. generate_mask() tiles that matrix to the token grid, and
agent_aware_attention() adds it to the logits before the softmax. That is
exactly the logit_neg_inf policy, so we do not patch the attention at all. We
write minus infinity into agent_mask and the softmax turns it into a weight of
exactly zero. The other weights of the row grow, because the softmax always
normalises to 1.

weight_zero. The additive mask stays at 0, so every logit survives. We patch
the module attribute `softmax` of third_party/AgentFormer/model/agentformer_lib
for the length of one predict() call. The wrapper calls the real softmax and
then multiplies the result by the tiled keep matrix. There is NO
renormalisation, so the row of the ego loses the removed mass and the other
weights keep their old value. The two policies therefore answer two different
questions, and a robustness check that compares them reads two real numbers.

The patch is undone in a finally block, so the module attribute is the real
torch.nn.functional.softmax again as soon as predict() returns.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path

import numpy as np

import config
from src.models import base

AGENTFORMER_ROOT = config.ROOT / "third_party" / "AgentFormer"

NEG_INF = float("-inf")

# The module names that CONTRACT.md section 3 allows in attention_edges.
ENCODER, DECODER, CROSS = "encoder", "decoder", "cross"


@contextlib.contextmanager
def _inside_agentformer():
    """Run a block with AgentFormer as the working directory.

    Their utils/config.py globs 'cfg/**/*.yml' and reads results_root_dir as a
    relative path, so it only works from their own repository root. The change
    of directory is always undone, even after an error.
    """
    if not AGENTFORMER_ROOT.exists():
        raise FileNotFoundError(
            f"{AGENTFORMER_ROOT} is missing. Run scripts/setup_agentformer.py first."
        )
    old = os.getcwd()
    root = str(AGENTFORMER_ROOT)
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        os.chdir(old)


def agentformer_lib():
    """Return the module that holds agent_aware_attention.

    The caller must already sit inside _inside_agentformer(), because the
    AgentFormer package only imports from its own repository root. The test
    module reads the module attribute `softmax` from here, so it can prove
    that the weight_zero patch is undone.
    """
    return importlib.import_module("model.agentformer_lib")


def _allow_full_pickle(torch_module) -> None:
    """Let torch.load read the released checkpoints.

    PyTorch 2.6 changed the default of weights_only to True. The AgentFormer
    checkpoints are full pickles, so they need the old default. The checkpoints
    come from a known URL and setup_agentformer.py checks their sha256.
    """
    if getattr(torch_module.load, "_capstone_patched", False):
        return
    original = torch_module.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    load._capstone_patched = True
    torch_module.load = load


class AgentFormerPredictor:
    """Wrap one released AgentFormer checkpoint behind the section 4.1 protocol."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        n_samples: int = config.N_SAMPLES,
    ) -> None:
        """Build the model and load the weights.

        `checkpoint` is a scene name such as "eth", or an AgentFormer config
        name such as "eth_agentformer". config.SCENE_TO_CHECKPOINT maps one to
        the other.
        """
        import torch

        name = str(checkpoint)
        self.cfg_name = config.SCENE_TO_CHECKPOINT.get(name, name)
        self.n_samples = int(n_samples)

        _allow_full_pickle(torch)
        torch.set_default_dtype(torch.float32)
        torch.set_grad_enabled(False)

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("cuda is not available. Pass device='cpu'.")
        self.device = torch.device(device)
        self._torch = torch

        with _inside_agentformer():
            from model.model_lib import model_dict
            from utils.config import Config

            cfg = Config(self.cfg_name, tmp=False, create_dirs=False)
            self.epoch = cfg.get_last_epoch()
            if self.epoch is None:
                raise FileNotFoundError(
                    f"no checkpoint under results/{self.cfg_name}/models. "
                    "Run scripts/setup_agentformer.py first."
                )
            model = model_dict[cfg.get("model_id", "agentformer")](cfg)
            model.set_device(self.device)
            model.eval()
            state = torch.load(cfg.model_path % self.epoch, map_location="cpu")
            model.load_state_dict(state["model_dict"], strict=False)

        self.cfg = cfg
        self.model = model
        self.traj_scale = float(cfg.traj_scale)
        self.n_hist = int(cfg.past_frames)
        self.n_fut = int(cfg.future_frames)
        if int(cfg.sample_k) != self.n_samples:
            raise ValueError(
                f"{self.cfg_name} was trained for sample_k={cfg.sample_k}, "
                f"but n_samples is {self.n_samples}. DLow cannot change K."
            )

    # -- input ------------------------------------------------------------

    def _make_data(self, hist: np.ndarray) -> dict:
        """Turn one window into the dict that AgentFormer.set_data() wants.

        The future is never read on the inference path, because mode='infer'
        skips the future encoder. We therefore pass zeros. The test
        test_future_is_ignored proves that claim instead of assuming it.
        """
        torch = self._torch
        n_agents = int(hist.shape[0])
        scaled = np.asarray(hist, dtype=np.float32) / self.traj_scale
        return {
            "pre_motion_3D": [torch.from_numpy(scaled[i].copy()) for i in range(n_agents)],
            "fut_motion_3D": [torch.zeros(self.n_fut, 2) for _ in range(n_agents)],
            "pre_motion_mask": [torch.ones(self.n_hist) for _ in range(n_agents)],
            "fut_motion_mask": [torch.ones(self.n_fut) for _ in range(n_agents)],
            "heading": None,
            "traj_scale": self.traj_scale,
        }

    def _apply_edge_mask(self, edge_mask: np.ndarray | None, mask_policy: str) -> None:
        """Write our edge mask into AgentFormer's own agent_mask.

        Entry (dst, src) of edge_mask is True when the edge from src into dst
        stays.

        Under logit_neg_inf a False entry becomes minus infinity, which the
        softmax turns into a weight of exactly zero.

        Under weight_zero every entry becomes 0, so no logit is touched. The
        softmax patch of _weight_zero_softmax() removes the weight after the
        softmax instead.
        """
        if mask_policy not in config.MASK_POLICIES:
            raise NotImplementedError(
                f"{mask_policy!r} is not supported. The two policies of "
                f"CONTRACT.md section 5.7 are {config.MASK_POLICIES}."
            )
        if edge_mask is None:
            return
        torch = self._torch
        keep = np.asarray(edge_mask, dtype=bool)
        if mask_policy == "logit_neg_inf":
            additive = np.where(keep, 0.0, NEG_INF).astype(np.float32)
        else:
            additive = np.zeros(keep.shape, dtype=np.float32)
        self.model.data["agent_mask"] = torch.from_numpy(additive).to(self.device)

    @contextlib.contextmanager
    def _weight_zero_softmax(self, edge_mask: np.ndarray | None, mask_policy: str):
        """Multiply every attention weight by the keep matrix, after the softmax.

        The caller must already sit inside _inside_agentformer().

        The wrapper replaces the module attribute `softmax` of
        agentformer_lib, which the two softmax calls of agent_aware_attention
        read at call time. The wrapper calls the real softmax, then tiles the
        (N, N) keep matrix onto the token grid and multiplies. It never
        renormalises, so the removed mass simply disappears.

        The token grid is time-major, that is token = time * N + agent, and
        generate_mask() tiles the (N, N) matrix with repeat(). The tile factors
        therefore differ per call: the encoder self attention is N*8 by N*8,
        the decoder self attention grows to N*12 by N*12 over the
        autoregressive steps, and the cross attention is N*12 by N*8. The
        wrapper reads both factors from the tensor at call time.

        The patch is always undone, even after an error.
        """
        if mask_policy != "weight_zero" or edge_mask is None:
            yield
            return

        torch = self._torch
        n_agents = int(np.asarray(edge_mask).shape[0])
        keep = np.asarray(edge_mask, dtype=bool).astype(np.float32)
        keep_tensor = torch.from_numpy(keep).to(self.device)

        library = agentformer_lib()
        original = library.softmax

        def wrapper(*args, **kwargs):
            weights = original(*args, **kwargs)
            if weights.dim() < 2:
                raise ValueError(
                    "the weight_zero patch expects an attention tensor of at "
                    f"least 2 axes, it got {tuple(weights.shape)}"
                )
            rows, cols = int(weights.shape[-2]), int(weights.shape[-1])
            if rows % n_agents != 0 or cols % n_agents != 0:
                raise ValueError(
                    "the weight_zero patch cannot tile the keep matrix onto an "
                    f"attention grid of {rows} by {cols} tokens for "
                    f"{n_agents} agents. Both axes must divide by the agent "
                    "count. A start token would break this rule."
                )
            tiled = keep_tensor.repeat(rows // n_agents, cols // n_agents)
            return weights * tiled

        library.softmax = wrapper
        try:
            yield
        finally:
            library.softmax = original

    # -- the protocol -----------------------------------------------------

    def predict(
        self,
        hist: np.ndarray,
        edge_mask: np.ndarray | None = None,
        draws: np.ndarray | None = None,
        mask_policy: str = config.PRIMARY_MASK_POLICY,
    ) -> np.ndarray:
        """Return (K, N, 12, 2) float64 in metres.

        `draws` is accepted so that the call matches the protocol, but the
        released DLow model is deterministic, so `draws` changes nothing. Read
        the module docstring, fact 2.

        `mask_policy` is one of config.MASK_POLICIES. Read the MASKING section
        of the module docstring for the difference between the two.
        """
        hist = base.check_history(hist, self.n_hist)
        n_agents = int(hist.shape[0])
        keep = base.check_edge_mask(edge_mask, n_agents)

        data = self._make_data(hist)
        with _inside_agentformer():
            self.model.set_data(data)
            self._apply_edge_mask(keep, mask_policy)
            with self._weight_zero_softmax(keep, mask_policy):
                out, _ = self.model.inference(mode="infer", sample_num=self.n_samples)

        pred = out.transpose(0, 1).contiguous() * self.traj_scale
        pred = pred.detach().cpu().numpy().astype(np.float64)
        return base.check_prediction(pred, n_agents, self.n_fut)

    def attention(
        self, hist: np.ndarray, edge_mask: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Return module -> the attention map, agent-major on both axes.

        encoder is (N*8, N*8). decoder is (N*12, N*12). cross is (N*12, N*8).
        Every map is the mean over the layers. The heads are already averaged
        inside agent_aware_attention, which divides the head sum by the head
        count before it returns.
        """
        hist = base.check_history(hist, self.n_hist)
        n_agents = int(hist.shape[0])
        base.check_edge_mask(edge_mask, n_agents)

        captured: dict[str, list[np.ndarray]] = {ENCODER: [], DECODER: [], CROSS: []}
        handles = []

        def grab(name: str):
            def hook(_module, _inputs, output):
                weights = output[1]
                if weights is not None:
                    captured[name].append(weights[0].detach().float().cpu().numpy())

            return hook

        core = self._core_model()
        for layer in core.context_encoder.tf_encoder.layers:
            handles.append(layer.self_attn.register_forward_hook(grab(ENCODER)))
        for layer in core.future_decoder.tf_decoder.layers:
            handles.append(layer.self_attn.register_forward_hook(grab(DECODER)))
            handles.append(layer.multihead_attn.register_forward_hook(grab(CROSS)))

        data = self._make_data(hist)
        try:
            with _inside_agentformer():
                self.model.set_data(data)
                self._apply_edge_mask(edge_mask, config.PRIMARY_MASK_POLICY)
                self.model.inference(
                    mode="infer", sample_num=self.n_samples, need_weights=True
                )
        finally:
            for handle in handles:
                handle.remove()

        maps: dict[str, np.ndarray] = {}
        for name, stack in captured.items():
            if not stack:
                continue
            # The decoder is autoregressive, so it runs once per future step and
            # its sequence grows every time. Keep only the captures of the last
            # step, which hold the complete map, then average over the layers.
            widest = max(item.shape[0] for item in stack)
            full = [item for item in stack if item.shape[0] == widest]
            square = np.mean(np.stack(full, axis=0), axis=0)
            maps[name] = self._to_agent_major(square, n_agents)
        return maps

    # -- helpers ----------------------------------------------------------

    def _core_model(self):
        """Return the AgentFormer VAE, whether or not DLow wraps it."""
        return getattr(self.model, "pred_model", [self.model])[0]

    def _to_agent_major(self, square: np.ndarray, n_agents: int) -> np.ndarray:
        """Reorder a raw time-major map onto the agent-major axes.

        The two axes can carry different lengths, because the cross attention
        reads 12 future rows against 8 observed columns.
        """
        rows, cols = square.shape
        row_order = base.agent_major_permutation(n_agents, rows // n_agents)
        col_order = base.agent_major_permutation(n_agents, cols // n_agents)
        return square[np.ix_(row_order, col_order)].astype(np.float64)

    def __repr__(self) -> str:
        return (
            f"AgentFormerPredictor(cfg={self.cfg_name!r}, epoch={self.epoch}, "
            f"device={self.device}, K={self.n_samples})"
        )
