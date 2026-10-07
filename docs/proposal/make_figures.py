"""Figures for the P1 proposal PDF (numbers from experiments/role_swap/results/*.md)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # control / hard (validated pair)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def env_panels():
    from lewm_research.envs.pusht_peg import PushTPeg
    frames = []
    for radius, seed in ((15, 3), (15, 11), (45, 3), (45, 11)):
        env = PushTPeg(with_target=False, peg_radius=radius, render_mode="rgb_array")
        env.reset(seed=seed, options={"peg_placement": "clutter"})
        frames.append(np.asarray(env.render())); env.close()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for ax, img, title in zip(axes, frames, ["peg radius 15", "peg radius 15", "peg radius 45", "peg radius 45"]):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8, color=INK2)
        for s in ax.spines.values(): s.set_color(GRID)
    fig.tight_layout(); fig.savefig(OUT / "env.pdf"); plt.close(fig)


ROWS = ["Reference planner\n(true physics)", "Pretrained LeWM", "Peg-as-clutter, seed 0",
        "Peg-as-clutter, seed 1", "Peg-pushed (mixed), seed 0", "Peg-pushed (mixed), seed 1"]
# (point, lo, hi) on the reference-feasible common sets of the main N=400 run
G = {"hard": [(.989, .971, .996), (.008, 0, .020), (.017, .006, .031), (.011, .003, .023), (.025, .011, .042), (.006, 0, .014)],
     "ctrl": [(.992, .975, .997), (.487, .439, .538), (.705, .657, .751), (.739, .694, .782), (.691, .643, .739), (.671, .623, .722)]}
D = {"hard": [(.993, .975, .998), (.295, .247, .351), (.427, .372, .483), (.448, .392, .507), (.441, .382, .500), (.399, .347, .458)],
     "ctrl": [(.997, .981, .999), (.330, .278, .385), (.556, .500, .615), (.569, .514, .625), (.517, .462, .576), (.497, .438, .556)]}


def success_panel(ax, data, hard_label, ctrl_label, title):
    y = np.arange(len(ROWS))[::-1]; h = 0.36
    for key, color, off, label in (("ctrl", BLUE, h / 2 + 0.02, ctrl_label), ("hard", ORANGE, -h / 2 - 0.02, hard_label)):
        pts = np.array(data[key]) * 100
        ax.barh(y + off, pts[:, 0], height=h, color=color, label=label, zorder=2)
        ax.errorbar(pts[:, 0], y + off, xerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]],
                    fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
        for xv, hi, yv in zip(pts[:, 0], pts[:, 2], y + off):
            ax.text(hi + 2, yv, f"{xv:.0f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(ROWS, fontsize=8, color=INK)
    ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("planning success (%)")
    ax.grid(axis="x", color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=22)
    ax.legend(loc="lower left", fontsize=7, frameon=False, bbox_to_anchor=(0.0, 1.0), ncol=2,
              handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)


def success_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), sharey=True)
    success_panel(axes[0], G, "move the peg", "move the T", "(a) Target role (353 scenes)")
    success_panel(axes[1], D, "peg beside path", "peg away from path", "(b) Don't-disturb role (288 scenes)")
    fig.tight_layout(); fig.savefig(OUT / "success.pdf"); plt.close(fig)


READ = [("Pretrained LeWM", 164.8, 25.7), ("Peg-as-clutter, s0", 161.2, 14.5), ("Peg-as-clutter, s1", 162.9, 14.9),
        ("Peg-pushed, s0", 164.1, 17.5), ("Peg-pushed, s1", 164.1, 17.8), ("Peg-pushed, 5x longer", 163.3, 16.0),
        ("+ inverse-dynamics loss", 107.2, 33.2), ("+ peg-position labels", 19.2, 14.4)]


def readout_figure():
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    y = np.arange(len(READ))[::-1]
    ax.axvline(186.1, color=INK2, ls="--", lw=1); ax.text(186.1, len(READ) - 0.35, "guessing the\naverage (peg)", fontsize=7, color=INK2, ha="center", va="bottom")
    ax.axvline(19.5, color=INK2, ls=":", lw=1); ax.text(19.5, len(READ) - 0.35, "raw pixels\n(peg)", fontsize=7, color=INK2, ha="center", va="bottom")
    for (name, peg, t), yy in zip(READ, y):
        ax.plot([t, peg], [yy, yy], color=GRID, lw=2, zorder=1)
    ax.scatter([r[2] for r in READ], y, s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3, label="T position error")
    ax.scatter([r[1] for r in READ], y, s=40, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3, label="peg position error")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in READ], fontsize=8, color=INK)
    ax.set_xlim(0, 200); ax.set_ylim(-0.7, len(READ) + 0.4); ax.set_xlabel("readout error from the model's 192-number embedding (pixels; lower = better)")
    ax.grid(axis="x", color=GRID, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "readout.pdf"); plt.close(fig)


if __name__ == "__main__":
    env_panels(); success_figure(); readout_figure()
