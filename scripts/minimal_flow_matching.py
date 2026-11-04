"""Minimal flow matching on two moons, plus one reflow round.

Pure PyTorch, no dependency on the flow_matching library. Produces:
  figures/m0_two_moons.png      samples after 1, 2, 8, 32 Euler steps (vanilla model)
  figures/m0_reflow.png         one-step samples: vanilla vs after one reflow round

Usage:
  python scripts/minimal_flow_matching.py [--iters 10000] [--reflow-iters 10000] [--device cpu]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch import Tensor, nn

try:
    from sklearn.datasets import make_moons
except ImportError as e:  # pragma: no cover
    raise SystemExit("scikit-learn is required: pip install scikit-learn") from e


class Flow(nn.Module):
    """Velocity field v_theta(x_t, t) as a small MLP."""

    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), dim=-1))


def sample_data(n: int, device: torch.device) -> Tensor:
    x, _ = make_moons(n, noise=0.15)  # uses numpy's global RNG, seeded in train_models
    return torch.tensor(x, dtype=torch.float32, device=device)


def train(flow: Flow, pairs, iters: int, lr: float = 1e-2, batch: int = 256) -> list[float]:
    """Conditional flow matching. `pairs` is a callable returning (x_0, x_1) batches."""
    opt = torch.optim.Adam(flow.parameters(), lr)
    losses = []
    for _ in range(iters):
        x_0, x_1 = pairs(batch)
        t = torch.rand(len(x_1), 1, device=x_1.device)
        x_t = (1 - t) * x_0 + t * x_1          # conditional OT path
        dx_t = x_1 - x_0                       # its constant velocity
        loss = ((flow(t, x_t) - dx_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def euler_sample(flow: Flow, x_0: Tensor, n_steps: int, return_path: bool = False) -> Tensor:
    """Integrate dx = v(x, t) dt from t=0 to 1. With return_path, returns (n_steps+1, N, dim)."""
    x = x_0
    path = [x]
    ts = torch.linspace(0, 1, n_steps + 1, device=x_0.device)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        x = x + (t1 - t0) * flow(t0.expand(len(x), 1), x)
        path.append(x)
    return torch.stack(path) if return_path else x


@torch.no_grad()
def straightness(flow: Flow, device: torch.device, n: int = 4000) -> float:
    """Mean squared gap between the field along its own trajectories and the chord velocity."""
    x0 = torch.randn(n, 2, device=device)
    x1 = euler_sample(flow, x0, n_steps=100)
    total = 0.0
    for t in torch.linspace(0.05, 0.95, 10, device=device):
        xt = (1 - t) * x0 + t * x1
        total += ((flow(t.expand(n, 1), xt) - (x1 - x0)) ** 2).mean().item()
    return total / 10


def train_models(iters: int = 10_000, reflow_iters: int = 10_000, device: str = "cpu", seed: int = 0) -> dict:
    """Train the vanilla model and one reflow round. Returns models and metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    # 1) vanilla flow matching: independent coupling (x_0 ~ N(0, I), x_1 ~ data)
    flow = Flow().to(dev)

    def independent_pairs(b: int):
        x_1 = sample_data(b, dev)
        return torch.randn_like(x_1), x_1

    t0 = time.time()
    losses = train(flow, independent_pairs, iters)
    print(f"vanilla: {iters} iters in {time.time() - t0:.1f}s, "
          f"loss {sum(losses[:100]) / 100:.3f} -> {sum(losses[-100:]) / 100:.3f}")

    # 2) reflow: pairs produced by simulating the trained model
    z_0 = torch.randn(20_000, 2, device=dev)
    z_1 = euler_sample(flow, z_0, n_steps=100)
    flow_reflow = Flow().to(dev)

    def reflow_pairs(b: int):
        idx = torch.randint(0, len(z_0), (b,), device=dev)
        return z_0[idx], z_1[idx]

    t0 = time.time()
    losses_r = train(flow_reflow, reflow_pairs, reflow_iters)
    print(f"reflow:  {reflow_iters} iters in {time.time() - t0:.1f}s, "
          f"loss {sum(losses_r[:100]) / 100:.3f} -> {sum(losses_r[-100:]) / 100:.3f}")

    # 3) straightness: how far the learned field is from constant along its own trajectories
    s_v, s_r = straightness(flow, dev), straightness(flow_reflow, dev)
    print(f"straightness (lower is straighter): vanilla {s_v:.4f}, reflow {s_r:.4f}")

    return {
        "flow": flow, "flow_reflow": flow_reflow, "device": dev,
        "straightness_vanilla": s_v, "straightness_reflow": s_r,
        "loss_vanilla": sum(losses[-100:]) / 100, "loss_reflow": sum(losses_r[-100:]) / 100,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=10_000)
    p.add_argument("--reflow-iters", type=int, default=10_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = train_models(args.iters, args.reflow_iters, args.device, args.seed)
    flow, flow_reflow, device = out["flow"], out["flow_reflow"], out["device"]
    s_v, s_r = out["straightness_vanilla"], out["straightness_reflow"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_0 = torch.randn(1000, 2, device=device)
    steps = [1, 2, 8, 32]
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(4 * (len(steps) + 1), 4), sharex=True, sharey=True)
    data = sample_data(1000, device).cpu().numpy()
    axes[0].scatter(data[:, 0], data[:, 1], s=6, c="#888888")
    axes[0].set_title("data")
    for ax, n in zip(axes[1:], steps):
        x = euler_sample(flow, x_0, n).cpu().numpy()
        ax.scatter(x[:, 0], x[:, 1], s=6)
        ax.set_title(f"vanilla, {n} Euler step{'s' if n > 1 else ''}")
    for ax in axes:
        ax.set_xlim(-2.5, 3.0)
        ax.set_ylim(-2.0, 2.5)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "m0_two_moons.png"), dpi=120)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].scatter(data[:, 0], data[:, 1], s=6, c="#888888")
    axes[0].set_title("data")
    x = euler_sample(flow, x_0, 1).cpu().numpy()
    axes[1].scatter(x[:, 0], x[:, 1], s=6)
    axes[1].set_title(f"vanilla, 1 step (straightness {s_v:.3f})")
    x = euler_sample(flow_reflow, x_0, 1).cpu().numpy()
    axes[2].scatter(x[:, 0], x[:, 1], s=6)
    axes[2].set_title(f"after 1 reflow, 1 step (straightness {s_r:.4f})")
    for ax in axes:
        ax.set_xlim(-2.5, 3.0)
        ax.set_ylim(-2.0, 2.5)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "m0_reflow.png"), dpi=120)
    print(f"wrote {args.outdir}/m0_two_moons.png and {args.outdir}/m0_reflow.png")


if __name__ == "__main__":
    main()
