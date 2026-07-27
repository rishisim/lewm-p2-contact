"""Project-owned fixed-depth PushT CEM cost-model adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from pusht_refiner import (
    LATENT_DIM,
    MAX_HISTORY,
    MaskedStagewiseRefiner,
    OperationLedger,
    SUPPORTED_DEPTHS,
    prefix_mask,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PushTRefinedCostModel(nn.Module):
    """Keep CEM unchanged while refining each autoregressive base prediction."""

    def __init__(
        self,
        base: nn.Module,
        refiner: MaskedStagewiseRefiner | None,
        refinement_depth: int,
    ) -> None:
        super().__init__()
        if isinstance(refinement_depth, bool) or refinement_depth not in SUPPORTED_DEPTHS:
            raise ValueError(f"unsupported fixed refinement depth {refinement_depth}")
        if refinement_depth > 0 and refiner is None:
            raise ValueError("positive depth requires a verified refiner")
        self.base = base
        self.refiner = refiner
        self.refinement_depth = refinement_depth
        self.ledger = OperationLedger()

    @classmethod
    def from_checkpoint(
        cls,
        base: nn.Module,
        checkpoint: Path,
        *,
        expected_sha256: str,
        refinement_depth: int,
    ) -> "PushTRefinedCostModel":
        observed = sha256(checkpoint)
        if observed != expected_sha256:
            raise RuntimeError(
                f"refiner checkpoint hash mismatch: expected {expected_sha256}, got {observed}"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        refiner = MaskedStagewiseRefiner(verified_export=True)
        result = refiner.load_state_dict(payload["state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("strict refiner checkpoint load failed")
        refiner.eval().requires_grad_(False)
        return cls(base, refiner, refinement_depth)

    def _refine(self, history: Tensor, actions: Tensor, base: Tensor) -> Tensor:
        rows, length = history.shape[:2]
        padded_history = torch.zeros(
            rows, MAX_HISTORY, LATENT_DIM, device=history.device, dtype=history.dtype
        )
        padded_actions = torch.zeros(
            rows, MAX_HISTORY, actions.shape[-1], device=actions.device, dtype=actions.dtype
        )
        padded_history[:, -length:] = history
        padded_actions[:, -length:] = actions
        mask = prefix_mask(
            torch.full((rows,), length, dtype=torch.int64, device=history.device)
        )
        assert self.refiner is not None
        return self.refiner(
            padded_history,
            padded_actions,
            mask,
            base,
            self.refinement_depth,
            ledger=self.ledger,
        )

    def rollout(self, info: dict, action_sequence: Tensor) -> dict:
        history_limit = MAX_HISTORY
        observation_history = info["pixels"].size(2)
        batch, samples, horizon = action_sequence.shape[:3]
        initial_actions, future_actions = torch.split(
            action_sequence,
            [observation_history, horizon - observation_history],
            dim=2,
        )
        initial = {
            key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)
        }
        initial["action"] = initial_actions[:, 0]
        initial = self.base.encode(initial)
        embedding = initial["emb"].unsqueeze(1).expand(batch, samples, -1, -1)
        embedding = rearrange(embedding, "b s ... -> (b s) ...").clone()
        actions = rearrange(initial_actions, "b s ... -> (b s) ...")
        future_actions = rearrange(future_actions, "b s ... -> (b s) ...")

        for transition in range(horizon):
            latent_history = embedding[:, -history_limit:]
            action_history = actions[:, -history_limit:]
            encoded_action = self.base.action_encoder(action_history)
            base_prediction = self.base.predict(
                latent_history, encoded_action
            )[:, -1]
            if self.refinement_depth == 0:
                self.ledger.record_base(len(base_prediction))
                prediction = base_prediction
            else:
                prediction = self._refine(
                    latent_history, action_history, base_prediction
                )
            embedding = torch.cat([embedding, prediction[:, None]], dim=1)
            if transition < horizon - observation_history:
                actions = torch.cat(
                    [actions, future_actions[:, transition : transition + 1]], dim=1
                )

        info["predicted_emb"] = rearrange(
            embedding, "(b s) ... -> b s ...", b=batch, s=samples
        )
        return info

    @torch.inference_mode()
    def get_cost(self, info_dict: dict, action_candidates: Tensor) -> Tensor:
        batch, samples = map(int, action_candidates.shape[:2])
        self.ledger.record_goal_encoder(batch)
        self.ledger.record_image_encoder(batch)
        self.ledger.record_terminal_cost(batch * samples)
        if self.refinement_depth == 0:
            rows = batch * samples
            for _ in range(int(action_candidates.shape[2])):
                self.ledger.record_base(rows)
            return self.base.get_cost(info_dict, action_candidates)
        device = next(self.parameters()).device
        local = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in info_dict.items()
        }
        goal = {key: value[:, 0] for key, value in local.items() if torch.is_tensor(value)}
        goal["pixels"] = goal["goal"]
        for key in list(goal):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        goal_embedding = self.base.encode(goal)["emb"]
        local["goal_emb"] = goal_embedding
        local = self.rollout(local, action_candidates.to(device))
        predicted = local["predicted_emb"][..., -1:, :]
        expanded_goal = goal_embedding.unsqueeze(1)[..., -1:, :].expand_as(predicted)
        return F.mse_loss(
            predicted, expanded_goal.detach(), reduction="none"
        ).sum(dim=tuple(range(2, predicted.ndim)))
