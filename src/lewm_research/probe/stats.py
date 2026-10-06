"""Deterministic descriptive statistics, with bases as sampling units."""

from statistics import NormalDist
import numpy as np
from scipy.stats import spearmanr


def wilson_ci(successes, total, confidence=0.95):
    if total < 1 or successes < 0 or successes > total or not 0 < confidence < 1:
        raise ValueError("invalid binomial counts or confidence")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, denom = successes / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    width = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denom
    return float(max(0, center - width)), float(min(1, center + width))


def paired_cluster_bootstrap(left, right, base_ids, seed=0, samples=2000):
    """Mean paired difference; resample whole bases, retaining all their rows."""
    left, right, ids = np.asarray(left, float), np.asarray(right, float), np.asarray(base_ids)
    if left.shape != right.shape or left.ndim != 1 or ids.shape != left.shape or not len(left) or samples < 1:
        raise ValueError("aligned nonempty paired rows required")
    groups = np.unique(ids)
    differences = left - right
    sums = np.array([differences[ids == g].sum() for g in groups])
    counts = np.array([(ids == g).sum() for g in groups])
    draws = np.random.default_rng(seed).integers(len(groups), size=(samples, len(groups)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return {"difference": float(differences.mean()), "ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            "bases": len(groups), "samples": samples, "seed": seed}


def spearman_bootstrap(x, y, seed=0, samples=2000, base_ids=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or samples < 1:
        raise ValueError("at least two aligned observations required")
    ids = np.arange(len(x)) if base_ids is None else np.asarray(base_ids)
    if ids.shape != x.shape:
        raise ValueError("base IDs must align")
    groups = [np.flatnonzero(ids == g) for g in np.unique(ids)]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):
        rows = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if np.ptp(x[rows]) and np.ptp(y[rows]):
            boot.append(float(spearmanr(x[rows], y[rows]).statistic))
    rho = float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else None
    return {"rho": rho, "ci": np.quantile(boot, [0.025, 0.975]).tolist() if boot else None,
            "valid_bootstraps": len(boot)}


def top1_regret(predicted, physical):
    predicted, physical = np.asarray(predicted), np.asarray(physical)
    if predicted.ndim != 1 or predicted.shape != physical.shape or not len(predicted):
        raise ValueError("aligned nonempty costs required")
    return float(physical[np.argmin(predicted)] - physical.min())


def paired_sample_size(discordant, total, effect=.15, power=.8):
    """Planning approximation for a paired binary CI excluding zero.

    q=P(discordance), Var(Y_hard-Y_control)=q-effect**2. Use q's
    Wilson upper bound because a 20-base estimate is imprecise. This powers
    significance only; it cannot guarantee the bar's effect-size/localization
    requirements or transfer from pretrained to fine-tuned checkpoints.
    """
    if total < 1 or not 0 < effect < 1 or not 0 < power < 1:
        raise ValueError("positive sample size, effect, and power required")
    q = max(effect, wilson_ci(discordant, total)[1])
    z = NormalDist().inv_cdf(.975) + NormalDist().inv_cdf(power)
    return {"discordance": discordant / total, "discordance_upper": q,
            "effect": effect, "power": power,
            "effective_bases": int(np.ceil(z**2 * (q-effect**2) / effect**2))}
