"""Smoke test for flow_matching-main/examples/text on a laptop.

The real entry point (run_train.py) needs a SLURM cluster, 8 GPUs, wandb and the FineWeb-EDU dataset.
This script exercises the same building blocks on a tiny synthetic task so we can check that the
model, path, both losses, both source distributions and the sampler all run and actually learn:

  input   sequences of 16 tokens from a vocabulary of 64 that count upward by a fixed step of
          1, 2 or 3 (mod 64), e.g. 10 12 14 16 ...
  model   the example's Transformer (2 blocks, hidden 64) from examples/text/model
  train   the example's path / source / loss objects from examples/text/logic/flow.py, using the
          same lines as logic/training.py:98-114
  sample  the example's generate_samples() from logic/generate.py (MixtureDiscreteEulerSolver)
  output  "pattern accuracy": fraction of adjacent token pairs whose difference is the sequence's
          own most common step in {1, 2, 3}. Chance level is about 5 percent.

Usage:
  python scripts/smoke_test_text_example.py [--steps 400]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "flow_matching-main"))
sys.path.insert(0, os.path.join(ROOT, "flow_matching-main", "examples", "text"))

import torch  # noqa: E402
from flow_matching.loss import MixturePathGeneralizedKL  # noqa: E402
from logic import flow  # noqa: E402  (examples/text/logic/flow.py)
from logic.generate import generate_samples  # noqa: E402
from model import Transformer  # noqa: E402  (examples/text/model/transformer.py)
from torch import nn  # noqa: E402

VOCAB, LENGTH = 64, 16
MODEL_CFG = dict(hidden_size=64, cond_dim=32, length=LENGTH, n_blocks=2, n_heads=4, dropout=0.0)


def sample_data(batch: int) -> torch.Tensor:
    start = torch.randint(0, VOCAB, (batch, 1))
    step = torch.randint(1, 4, (batch, 1))
    return (start + step * torch.arange(LENGTH)[None]) % VOCAB


def pattern_accuracy(x: torch.Tensor) -> float:
    x = x.clamp(max=VOCAB - 1)                       # a leftover mask token counts as wrong
    diff = (x[:, 1:] - x[:, :-1]) % VOCAB
    best = torch.stack([(diff == s).float().mean(dim=1) for s in (1, 2, 3)]).max(dim=0).values
    return best.mean().item()


class _FakeTokenizer:
    def batch_decode(self, sample):
        return [" ".join(str(int(t)) for t in row) for row in sample]


def run(source: str, loss_name: str, steps: int, device: torch.device) -> dict:
    torch.manual_seed(0)
    source_distribution = flow.get_source_distribution(source, vocab_size=VOCAB)
    path = flow.get_path(scheduler_type="polynomial", exponent=1.0)
    loss_fn = flow.get_loss_function(loss_name, path=path)
    time_epsilon = 1e-3 if isinstance(loss_fn, MixturePathGeneralizedKL) else 0.0

    model = Transformer(vocab_size=VOCAB, masked=source_distribution.masked, config=MODEL_CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    t0, first, last = time.time(), None, None
    model.train()
    for i in range(steps):
        x_1 = sample_data(64).to(device)
        with torch.no_grad():                                        # logic/training.py:98-101
            x_0 = source_distribution.sample_like(x_1)
            t = torch.rand(x_1.shape[0], device=device) * (1.0 - time_epsilon)
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        logits = model(x_t=path_sample.x_t, time=path_sample.t)       # logic/training.py:106-114
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
        else:
            loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i < 20:
            first = loss.item() if first is None else 0.9 * first + 0.1 * loss.item()
        last = loss.item() if last is None else 0.9 * last + 0.1 * loss.item()
    train_s = time.time() - t0

    model.eval()
    sample = generate_samples(
        model=model, step=steps, vocab_size=VOCAB, tokenizer=_FakeTokenizer(), rank=0, device=device,
        path=path, source_distribution=source_distribution, sample_batch_size=256,
        sequence_length=LENGTH, sampling_steps=32, time_epsilon=time_epsilon, sample_dir=None,
    )
    return {"source": source, "loss": loss_name, "loss_first": first, "loss_last": last,
            "train_s": train_s, "acc": pattern_accuracy(sample.cpu()), "example": sample[0].tolist()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()
    device = torch.device("cpu")

    print(f"data example: {sample_data(1)[0].tolist()}")
    print(f"chance-level pattern accuracy (random tokens): "
          f"{pattern_accuracy(torch.randint(0, VOCAB, (4096, LENGTH))):.3f}\n")
    results = []
    for source in ("uniform", "mask"):
        for loss_name in ("cross_entropy", "generalized_kl"):
            r = run(source, loss_name, args.steps, device)
            results.append(r)
            print(f"[{'PASS' if r['acc'] > 0.5 else 'WEAK'}] source={source:8s} loss={loss_name:15s} "
                  f"loss {r['loss_first']:.3f} -> {r['loss_last']:.3f}  pattern acc {r['acc']:.3f}  "
                  f"({r['train_s']:.0f}s)\n         sample: {r['example']}", flush=True)
    ok = all(r["acc"] > 0.5 for r in results)
    print(f"\n{'ALL PASS' if ok else 'SOME WEAK'}: {sum(r['acc'] > 0.5 for r in results)}/{len(results)} configurations learned the pattern")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
