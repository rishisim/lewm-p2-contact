#!/usr/bin/env python3
"""Stage 1b/1c comparison for Proposal 1.

Consumes the Stage 1a latent cache and reports persistence, history-only MLP,
and frozen action-conditioned LeWM next-latent errors.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[2]
for rel in ("stable-worldmodel-readonly", "stable-pretraining-readonly", "le-wm"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402


OUT_DIR = ROOT / "prelim-p1" / "outputs"
CACHE_PATH = OUT_DIR / "stage1a_latents.npz"
STAGE0_PATH = OUT_DIR / "stage0_timing.json"
STAGE1A_PATH = OUT_DIR / "stage1a_summary.json"

SEED = 123
HISTORY_SIZE = 3
LATENT_DIM = 192
ACTION_BLOCK_DIM = 10
HISTORY_EPOCHS = 2000
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4


class HistoryOnlyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HISTORY_SIZE * LATENT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, LATENT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def metric_row(pred: np.ndarray, target: np.ndarray, persistence_mse: float | None = None) -> dict:
    err2 = (pred - target) ** 2
    abs_mse = float(err2.mean())
    var_dim = target.var(axis=0, ddof=0)
    denom = float(var_dim.mean())
    normalized = abs_mse / max(denom, 1e-12)
    row = {
        "abs_mse": abs_mse,
        "normalized_mse": normalized,
    }
    if persistence_mse is not None:
        row["fraction_of_persistence"] = abs_mse / max(persistence_mse, 1e-12)
    return row


def format_float(x: float) -> str:
    return f"{x:.6f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def train_history_probe(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_test_np: np.ndarray,
    y_test_np: np.ndarray,
    device: torch.device,
) -> tuple[HistoryOnlyMLP, dict, list[dict]]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = HistoryOnlyMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    x_train = torch.from_numpy(x_train_np).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_test = torch.from_numpy(x_test_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)

    history = []
    n = x_train.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    start_time = time.perf_counter()
    for epoch in range(1, HISTORY_EPOCHS + 1):
        perm = torch.randperm(n, generator=generator)
        model.train()
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x_train[idx])
            loss = loss_fn(pred, y_train[idx])
            loss.backward()
            opt.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == HISTORY_EPOCHS:
            model.eval()
            with torch.no_grad():
                train_loss = loss_fn(model(x_train), y_train).item()
                test_loss = loss_fn(model(x_test), y_test).item()
            history.append({"epoch": epoch, "train_mse": train_loss, "test_mse": test_loss})

    sync(device)
    elapsed = time.perf_counter() - start_time
    model.eval()
    with torch.no_grad():
        final_train = loss_fn(model(x_train), y_train).item()
        final_test = loss_fn(model(x_test), y_test).item()
    train_info = {
        "architecture": "576->512->192 ReLU MLP",
        "epochs": HISTORY_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "final_train_mse": float(final_train),
        "final_test_mse": float(final_test),
        "elapsed_seconds": elapsed,
    }
    return model, train_info, history


def predict_history_probe(model: HistoryOnlyMLP, x_np: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(x_np).to(device)
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], 512):
            outs.append(model(x[start : start + 512]).detach().cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)


def predict_full_model(history_np: np.ndarray, action_np: np.ndarray, device: torch.device) -> np.ndarray:
    model = swm.policy.AutoCostModel("pusht/lewm", cache_dir=swm.data.utils.get_cache_dir())
    model = model.to(device).eval()
    model.requires_grad_(False)

    preds = []
    with torch.no_grad():
        for start in range(0, history_np.shape[0], 512):
            hist = torch.from_numpy(history_np[start : start + 512]).to(device)
            action = torch.from_numpy(action_np[start : start + 512]).to(device)
            act_emb = model.action_encoder(action)
            pred = model.predict(hist, act_emb)[:, -1]
            preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0).numpy().astype(np.float32)


def subset_metrics(preds: dict[str, np.ndarray], target: np.ndarray) -> dict[str, dict]:
    persistence_mse = mse(preds["persistence"], target)
    rows = {}
    for name, pred in preds.items():
        rows[name] = metric_row(pred, target, persistence_mse)
    return rows


def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing Stage 1a cache: {CACHE_PATH}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    device = torch.device("mps")

    data = np.load(CACHE_PATH)
    history = data["history"].astype(np.float32)
    action = data["action"].astype(np.float32)
    target = data["target"].astype(np.float32)
    split = data["split"]
    contact = data["contact"]

    train_mask = split == 0
    test_mask = split == 1
    x_train = history[train_mask].reshape(train_mask.sum(), -1)
    y_train = target[train_mask]
    x_test = history[test_mask].reshape(test_mask.sum(), -1)
    y_test = target[test_mask]

    probe, probe_train_info, training_history = train_history_probe(
        x_train, y_train, x_test, y_test, device
    )

    test_history_pred = predict_history_probe(probe, x_test, device)
    train_history_pred = predict_history_probe(probe, x_train, device)
    test_persistence_pred = history[test_mask, -1]
    train_persistence_pred = history[train_mask, -1]
    test_full_pred = predict_full_model(history[test_mask], action[test_mask], device)
    train_full_pred = predict_full_model(history[train_mask], action[train_mask], device)

    test_preds = {
        "persistence": test_persistence_pred,
        "history_only": test_history_pred,
        "full_lewm": test_full_pred,
    }
    train_preds = {
        "persistence": train_persistence_pred,
        "history_only": train_history_pred,
        "full_lewm": train_full_pred,
    }

    core_test = subset_metrics(test_preds, y_test)
    core_train = subset_metrics(train_preds, y_train)
    action_info_value = (
        core_test["history_only"]["abs_mse"] - core_test["full_lewm"]["abs_mse"]
    ) / max(core_test["history_only"]["abs_mse"], 1e-12)
    headroom = (
        core_test["persistence"]["abs_mse"] - core_test["history_only"]["abs_mse"]
    ) / max(core_test["persistence"]["abs_mse"], 1e-12)

    test_contact = contact[test_mask] > 0
    contact_results = {}
    for subset_name, mask in {
        "non_contact": ~test_contact,
        "contact": test_contact,
    }.items():
        subset_preds = {k: v[mask] for k, v in test_preds.items()}
        subset_target = y_test[mask]
        if len(subset_target) == 0:
            contact_results[subset_name] = {"n": 0, "metrics": {}, "action_information_value": None}
            continue
        metrics = subset_metrics(subset_preds, subset_target)
        subset_aiv = (
            metrics["history_only"]["abs_mse"] - metrics["full_lewm"]["abs_mse"]
        ) / max(metrics["history_only"]["abs_mse"], 1e-12)
        contact_results[subset_name] = {
            "n": int(mask.sum()),
            "metrics": metrics,
            "action_information_value": float(subset_aiv),
        }

    results = {
        "stage": "1b_1c",
        "device": str(device),
        "probe_train": probe_train_info,
        "training_history": training_history,
        "train_metrics": core_train,
        "test_metrics": core_test,
        "headline": {
            "action_information_value": float(action_info_value),
            "predictive_headroom_over_persistence": float(headroom),
        },
        "contact_split": contact_results,
    }
    result_path = OUT_DIR / "stage1bc_results.json"
    result_path.write_text(json.dumps(results, indent=2))

    stage0 = json.loads(STAGE0_PATH.read_text()) if STAGE0_PATH.exists() else {}
    stage1a = json.loads(STAGE1A_PATH.read_text()) if STAGE1A_PATH.exists() else {}
    report_path = OUT_DIR / "stage1_report.md"
    report_path.write_text(make_report(stage0, stage1a, results))

    print("STAGE 1B/1C SUMMARY")
    print(f"history-only final train MSE: {probe_train_info['final_train_mse']:.6f}")
    print(f"history-only final test MSE: {probe_train_info['final_test_mse']:.6f}")
    for method in ("persistence", "history_only", "full_lewm"):
        row = core_test[method]
        print(
            f"{method}: abs_mse={row['abs_mse']:.6f}, "
            f"normalized={row['normalized_mse']:.6f}, "
            f"fraction_persistence={row['fraction_of_persistence']:.6f}"
        )
    print(f"action_information_value: {action_info_value:.6f}")
    print(f"predictive_headroom_over_persistence: {headroom:.6f}")
    for subset_name in ("non_contact", "contact"):
        subset = contact_results[subset_name]
        print(
            f"{subset_name}: n={subset['n']}, "
            f"action_information_value={subset['action_information_value']:.6f}"
        )
    print(f"wrote: {result_path}")
    print(f"wrote: {report_path}")


def make_report(stage0: dict, stage1a: dict, results: dict) -> str:
    timing = stage0.get("timing", {})
    stage0_rows = [
        ["Load/device", str(stage0.get("load_ok")), str(stage0.get("device"))],
        ["Param count", f"{stage0.get('param_count', 0):,}", f"{stage0.get('param_count_millions', 0):.2f}M"],
        ["Encode", format_float(timing.get("encode_ms_per_frame", float("nan"))), "ms/frame"],
        ["Predictor fwd", format_float(timing.get("predictor_fwd_ms", float("nan"))), "ms"],
        ["Predictor fwd+bwd+step", format_float(timing.get("predictor_fwd_bwd_step_ms", float("nan"))), "ms"],
    ]

    methods = [
        ("Persistence", "persistence"),
        ("History-only MLP", "history_only"),
        ("Full LeWM", "full_lewm"),
    ]
    metric_rows = []
    for label, key in methods:
        row = results["test_metrics"][key]
        metric_rows.append(
            [
                label,
                format_float(row["abs_mse"]),
                format_float(row["normalized_mse"]),
                format_float(row["fraction_of_persistence"]),
            ]
        )

    contact_rows = []
    for subset_label, subset_key in (("Non-contact", "non_contact"), ("Contact", "contact")):
        subset = results["contact_split"][subset_key]
        metrics = subset["metrics"]
        for label, key in methods:
            row = metrics[key]
            contact_rows.append(
                [
                    subset_label,
                    str(subset["n"]),
                    label,
                    format_float(row["abs_mse"]),
                    format_float(row["normalized_mse"]),
                    format_float(row["fraction_of_persistence"]),
                    format_float(subset["action_information_value"]) if key == "history_only" else "",
                ]
            )

    headline = results["headline"]
    aiv = headline["action_information_value"]
    headroom = headline["predictive_headroom_over_persistence"]
    if headroom < 0:
        reading = (
            "The history-only probe is worse than persistence on the held-out trajectories, "
            "despite fitting the probe-train pairs very closely. That makes the formal action "
            "information value hard to interpret by itself: it is large because the frozen full "
            "LeWM model is strong and the trained history-only probe overfits/fails to generalize, "
            "not because this run cleanly isolates only action signal."
        )
    elif abs(headroom) < 0.05:
        reading = (
            "The history-only probe barely improves over persistence on the held-out trajectories, "
            "so the comparison is partly degenerate: these latents are already quite predictable by "
            "copying the last latent."
        )
    else:
        reading = (
            "The history-only probe changes the error substantially relative to persistence, so the "
            "comparison contains nontrivial predictive signal beyond simply copying the last latent."
        )
    if aiv > 0:
        reading += (
            f" The frozen action-conditioned LeWM lowers held-out MSE by {100 * aiv:.1f}% relative "
            "to the history-only probe."
        )
    else:
        reading += (
            f" The frozen action-conditioned LeWM does not beat the history-only probe in this run "
            f"(action information value {100 * aiv:.1f}%)."
        )

    divergences = []
    divergences.extend(stage0.get("api_verification", {}).get("live_source_discrepancies", []))
    divergences.append(
        "The HDF5 dataset has no n_contacts column; contact/non-contact uses the prior geometric fallback from 7D state."
    )
    divergences.append(
        "Known confound: the history-only probe is trained on latents from an already action-conditioned trained encoder, so this is a proxy and likely a lower bound on action-free pretraining plausibility."
    )
    divergences.append(
        "No explicit MPS CPU fallback warnings appeared in the captured Stage 0 terminal log."
    )

    lines = [
        "# Stage 1 Report: Proposal 1 Preliminary Test",
        "",
        "## Stage 0 Timing + Fallback",
        "",
        markdown_table(["Item", "Value", "Unit/Notes"], stage0_rows),
        "",
        f"MPS fallback status: {stage0.get('mps_fallback', {}).get('script_observed_warning_status', 'not recorded')}",
        "",
        "## 1a Extraction Summary",
        "",
        f"- Episodes: {stage1a.get('num_episodes')} total; {stage1a.get('train_trajectory_count')} train trajectories, {stage1a.get('test_trajectory_count')} test trajectories.",
        f"- Pairs: {stage1a.get('train_pairs')} train, {stage1a.get('test_pairs')} test, {stage1a.get('total_pairs')} total.",
        f"- History latents: `{tuple(stage1a.get('history_shape', []))}`; actions: `{tuple(stage1a.get('action_history_shape', []))}`; targets: `{tuple(stage1a.get('target_shape', []))}`.",
        f"- Latent stats: mean(mean_dim)={stage1a.get('latent_stats', {}).get('target_per_dim_mean_mean'):.6f}, mean(std_dim)={stage1a.get('latent_stats', {}).get('target_per_dim_std_mean'):.6f}, any_nan={stage1a.get('latent_stats', {}).get('any_nan')}.",
        f"- Contact labels: {stage1a.get('contact', {}).get('source')} with {stage1a.get('contact', {}).get('positive_pairs')} contact pairs and {stage1a.get('contact', {}).get('non_contact_pairs')} non-contact pairs.",
        "",
        "## 1b Three-Way Comparison",
        "",
        f"History-only probe: {results['probe_train']['architecture']}, {results['probe_train']['epochs']} epochs, final train MSE {results['probe_train']['final_train_mse']:.6f}, final test MSE {results['probe_train']['final_test_mse']:.6f}.",
        "",
        markdown_table(
            ["Method", "Abs MSE", "Normalized MSE", "Fraction of Persistence"],
            metric_rows,
        ),
        "",
        f"- Action information value: `{aiv:.6f}`",
        f"- Predictive headroom over persistence: `{headroom:.6f}`",
        "",
        "## 1c Contact Split",
        "",
        markdown_table(
            [
                "Subset",
                "n",
                "Method",
                "Abs MSE",
                "Normalized MSE",
                "Fraction of Persistence",
                "Action Info Value",
            ],
            contact_rows,
        ),
        "",
        "## Plain-Language Reading",
        "",
        reading,
        "",
        "## Divergences, Caveats, Confounds",
        "",
    ]
    lines.extend(f"- {item}" for item in divergences)
    lines.append("")
    lines.append("Stage 2 was not run.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
