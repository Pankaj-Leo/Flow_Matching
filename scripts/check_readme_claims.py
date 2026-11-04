"""Reproduce the numbers and code claims stated in README.md.

  1. Section 13: the per-token `interpolate` matches CondOTProbPath.sample when all tokens share one time.
  2. Section 3: the closed-form marginal velocity for point-mass data matches a Monte Carlo estimate.
  3. Sections 5 and 6: on two moons, sliced W2 to the data by Euler step count, before and after one
     reflow, plus straightness. Uses scripts/minimal_flow_matching.py (about 20 s on CPU).

Usage:
  PYTHONPATH=flow_matching-main python scripts/check_readme_claims.py
Needs torch, numpy, matplotlib and torchdiffeq.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "flow_matching-main"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402
from flow_matching.path import CondOTProbPath  # noqa: E402
from flow_matching.path.scheduler import CondOTScheduler  # noqa: E402
from minimal_flow_matching import euler_sample, sample_data, train_models  # noqa: E402


def interpolate(x1, x0, t_tok, sched=CondOTScheduler()):
    s = sched(t_tok)
    shape = t_tok.shape + (1,) * (x1.ndim - t_tok.ndim)
    a, sg = s.alpha_t.reshape(shape), s.sigma_t.reshape(shape)
    da, dsg = s.d_alpha_t.reshape(shape), s.d_sigma_t.reshape(shape)
    return a * x1 + sg * x0, da * x1 + dsg * x0


def sliced_w2(a: torch.Tensor, b: torch.Tensor, k: int = 256) -> float:
    d = torch.randn(a.shape[1], k)
    d = d / d.norm(dim=0)
    pa, _ = torch.sort(a @ d, 0)
    pb, _ = torch.sort(b @ d, 0)
    return ((pa - pb) ** 2).mean().sqrt().item()


def main() -> None:
    torch.manual_seed(0)
    b, n, d = 8, 5, 3
    x1, x0, t = torch.randn(b, n, d), torch.randn(b, n, d), torch.rand(b)
    xt, target = interpolate(x1, x0, t[:, None].expand(b, n))
    ref = CondOTProbPath().sample(x_0=x0, x_1=x1, t=t)
    print(f"[1] interpolate == CondOTProbPath.sample: "
          f"{torch.allclose(xt, ref.x_t) and torch.allclose(target, ref.dx_t)}")

    mu, pi, x, tt = torch.tensor([-3.0, 0.5, 3.0]), torch.tensor([0.45, 0.2, 0.35]), 0.7, 0.6
    w = pi * torch.exp(-0.5 * ((x - tt * mu) / (1 - tt)) ** 2)
    closed = ((w / w.sum()) * (mu - x) / (1 - tt)).sum().item()
    idx = torch.multinomial(pi, 4_000_000, replacement=True)
    m1, m0 = mu[idx], torch.randn(len(idx))
    near = ((tt * m1 + (1 - tt) * m0) - x).abs() < 0.01
    print(f"[2] marginal velocity: closed form {closed:.3f}, Monte Carlo {(m1 - m0)[near].mean().item():.3f}")

    r = train_models(seed=0)
    dev, n_eval = r["device"], 10_000
    torch.manual_seed(1)
    data = sample_data(n_eval, dev)
    print(f"[3] sliced W2 to data (noise floor {sliced_w2(sample_data(n_eval, dev), data):.3f})")
    with torch.no_grad():
        for name, label in (("flow", "vanilla"), ("flow_reflow", "reflow ")):
            row = [f"{k} step(s) {sliced_w2(euler_sample(r[name], torch.randn(n_eval, 2, device=dev), k), data):.3f}"
                   for k in (1, 2, 4, 8, 64)]
            print(f"    {label}: " + ", ".join(row))


if __name__ == "__main__":
    main()
