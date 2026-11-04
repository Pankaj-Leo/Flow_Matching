"""Run the parts of AlphaFlow that need only PyTorch: the harmonic prior and Kabsch alignment.

The full model (alphaflow-master/train.py, predict.py) needs CUDA, OpenFold and AlphaFold/ESMFold
weights. These two pieces are the flow-matching-specific changes AlphaFlow makes to the noise side:

  harmonic prior   noise that already looks like a chain (RMS neighbour distance 3.8 A by construction), not white noise
  Kabsch alignment rotate and shift the noise onto the data before interpolating x_t

Usage:
  python scripts/alphaflow_prior_demo.py [--residues 64] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "alphaflow-master"))

import torch  # noqa: E402
from alphaflow.utils.diffusion import HarmonicPrior, kabsch_rmsd  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--residues", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    x = HarmonicPrior(args.residues).sample()
    white = torch.randn(args.residues, 3)
    gap = lambda c: (c[1:] - c[:-1]).norm(dim=-1).mean().item()  # noqa: E731
    print(f"neighbour distance, harmonic prior: {gap(x):.2f} A   white noise: {gap(white):.2f} A")

    rot, _ = torch.linalg.qr(torch.randn(3, 3))
    y = x @ rot.T + 5.0
    before = (x - y).norm(dim=-1).pow(2).mean().sqrt().item()
    print(f"RMSD of a rotated and shifted copy: {before:.3f} A before Kabsch, {kabsch_rmsd(x, y).item():.6f} A after")


if __name__ == "__main__":
    main()
