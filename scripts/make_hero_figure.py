"""Hero figure for the README and blog post: flow matching at a glance.

Three panels on two moons, all from the same 30-line training loop:
  1. training pairs: straight lines between randomly paired noise and data
  2. the learned flow: trajectories bend where training paths crossed
  3. after one reflow round: trajectories are nearly straight

Usage:
  python scripts/make_hero_figure.py            # writes FlowMatching.png in the repo root
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimal_flow_matching import euler_sample, sample_data, train_models  # noqa: E402

# Palette: validated two-hue categorical pair on a light surface (colorblind-safe all-pairs).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
NOISE = "#2a78d6"      # source samples x_0
DATA = "#eb6834"       # data / generated samples x_1
PATH = "#898781"       # trajectories (neutral, recessive)
BACKDROP = "#e6e4dc"   # faint cloud showing the data distribution


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="FlowMatching.png")
    p.add_argument("--n", type=int, default=44, help="number of trajectories per panel")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = train_models(seed=args.seed)
    flow, flow_reflow, device = out["flow"], out["flow_reflow"], out["device"]

    g = torch.Generator().manual_seed(7)
    x_0 = torch.randn(args.n, 2, generator=g).to(device)
    x_1_random = sample_data(args.n, device)                        # independent of x_0
    path_vanilla = euler_sample(flow, x_0, 64, return_path=True).cpu().numpy()
    path_reflow = euler_sample(flow_reflow, x_0, 64, return_path=True).cpu().numpy()
    x0 = x_0.cpu().numpy()
    x1r = x_1_random.cpu().numpy()
    backdrop = sample_data(2500, device).cpu().numpy()             # where the data lives

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.76, bottom=0.11, wspace=0.05)

    panels = [
        ("1. Train on straight lines",
         "Noise and data are paired at random,\nso the straight training paths cross."),
        ("2. The model learns the average",
         f"Where paths crossed, trajectories bend.\nStraightness {out['straightness_vanilla']:.3f}: needs about 8 Euler steps."),
        ("3. Reflow straightens the flow",
         f"Retrained on its own (noise, sample) pairs.\nStraightness {out['straightness_reflow']:.4f}: one Euler step is enough."),
    ]

    def dots(ax, pts, color):
        # markers >= 8 px with a surface-colored ring so overlaps stay legible
        ax.scatter(pts[:, 0], pts[:, 1], s=46, c=color, edgecolors=SURFACE, linewidths=1.2, zorder=3)

    # panel 1: straight conditional paths under the independent coupling
    ax = axes[0]
    for a, b in zip(x0, x1r):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=PATH, lw=1.1, alpha=0.75, zorder=1, solid_capstyle="round")
    dots(ax, x0, NOISE)
    dots(ax, x1r, DATA)

    # panels 2 and 3: trajectories of the learned fields from the same noise points
    for ax, path in zip(axes[1:], (path_vanilla, path_reflow)):
        for i in range(path.shape[1]):
            ax.plot(path[:, i, 0], path[:, i, 1], color=PATH, lw=1.1, alpha=0.75, zorder=1, solid_capstyle="round")
        dots(ax, path[0], NOISE)
        dots(ax, path[-1], DATA)

    for ax, (title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.scatter(backdrop[:, 0], backdrop[:, 1], s=7, c=BACKDROP, linewidths=0, zorder=0)
        ax.set_xlim(-3.3, 3.2)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e1e0d9")
            s.set_linewidth(1.0)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK, pad=40)
        ax.text(0.0, 1.025, sub, transform=ax.transAxes, fontsize=11.5, color=INK_2, va="bottom", linespacing=1.35)

    fig.suptitle("Flow Matching at a glance", x=0.02, y=0.975, ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, "Two moons in 2D. One 30-line training loop, a velocity field, and an ODE.",
             fontsize=13, color=INK_2, ha="left")

    handles = [
        Line2D([], [], marker="o", ls="", color=NOISE, markeredgecolor=SURFACE, markersize=9, label="noise  x₀ ~ N(0, I)"),
        Line2D([], [], marker="o", ls="", color=DATA, markeredgecolor=SURFACE, markersize=9, label="data / generated sample  x₁"),
        Line2D([], [], color=PATH, lw=1.6, label="path from t = 0 to t = 1"),
        Line2D([], [], marker="o", ls="", color=BACKDROP, markersize=9, label="data distribution"),
    ]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.012), ncol=4, frameon=False,
                     fontsize=12, handletextpad=0.5, columnspacing=2.2)
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
