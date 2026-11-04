"""Execute the flow_matching example notebooks as plain scripts and report what ran.

Each notebook's code cells are concatenated, figures are saved instead of shown, and the script is
run in a subprocess from flow_matching-main/examples with the library on PYTHONPATH.

  --mode smoke   (default) shrink iteration counts and sample sizes so every notebook finishes in
                 seconds; proves the code paths execute on this machine
  --mode full    run the notebooks unmodified (minutes each on CPU)

Usage:
  python scripts/run_notebooks.py [--mode smoke|full] [--only SUBSTRING] [--timeout 3600]
Outputs:
  figures/notebooks/<mode>/<notebook>_<k>.png   every figure the notebook draws
  figures/notebooks/<mode>/summary.json         status, seconds, and the tail of stdout per notebook
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "flow_matching-main")
EXAMPLES = os.path.join(LIB, "examples")

NOTEBOOKS = [
    "standalone_flow_matching.ipynb",
    "standalone_discrete_flow_matching.ipynb",
    "2d_flow_matching.ipynb",
    "2d_discrete_flow_matching.ipynb",
    "2d_riemannian_flow_matching_sphere.ipynb",
    "2d_riemannian_flow_matching_flat_torus.ipynb",
    "2d_cnf_maximum_likelihood.ipynb",
]

# (regex, replacement) applied to notebook code in smoke mode only.
SMOKE_PATCHES = [
    (r"iterations = \d+", "iterations = 301"),
    (r"print_every = \d+", "print_every = 100"),
    (r"for _ in range\(10000\):", "for _ in range(300):"),
    (r"for i in range\(2000\):", "for i in range(15):"),          # CNF maximum-likelihood steps
    (r"batch_size = 4096", "batch_size = 512"),
    (r"batch_size = 50000", "batch_size = 2000"),
    (r"batch_size=50000", "batch_size=2000"),
    (r"n_samples = 1000000", "n_samples = 20000"),
    (r"grid_size = 200", "grid_size = 30"),
    (r"num_acc = 10", "num_acc = 2"),
    (r"n_discretization = 1024", "n_discretization = 16"),
]

PREAMBLE = '''
import os as _os
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
_FIG_DIR, _FIG_NAME, _fig_n = {fig_dir!r}, {name!r}, [0]
def _flush_figures(*_a, **_k):
    for _num in _plt.get_fignums():
        _fig_n[0] += 1
        _plt.figure(_num).savefig(_os.path.join(_FIG_DIR, f"{{_FIG_NAME}}_{{_fig_n[0]}}.png"), dpi=80, bbox_inches="tight")
    _plt.close("all")
_plt.show = _flush_figures
'''


def notebook_to_script(path: str, mode: str, fig_dir: str, name: str) -> tuple[str, list[str]]:
    cells = [c for c in json.load(open(path))["cells"] if c["cell_type"] == "code"]
    applied: list[str] = []
    parts = [PREAMBLE.format(fig_dir=fig_dir, name=name)]
    for i, cell in enumerate(cells):
        src = "".join(cell["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        if mode == "smoke":
            for pattern, repl in SMOKE_PATCHES:
                src, n = re.subn(pattern, repl, src)
                if n:
                    applied.append(repl)
        parts.append(f"# ---- cell {i} ----\n{src}\n_flush_figures()\n")
    return "\n".join(parts), applied


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--only", default="")
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()

    fig_dir = os.path.join(ROOT, "figures", "notebooks", args.mode)
    os.makedirs(fig_dir, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=LIB, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", PYTHONWARNINGS="ignore")

    summary = []
    for nb in NOTEBOOKS:
        if args.only and args.only not in nb:
            continue
        name = nb.replace(".ipynb", "")
        script, applied = notebook_to_script(os.path.join(EXAMPLES, nb), args.mode, fig_dir, name)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp = f.name
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, tmp], cwd=EXAMPLES, env=env, capture_output=True, text=True,
                               timeout=args.timeout)
            ok, out, err = r.returncode == 0, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            ok, out, err = False, (e.stdout or ""), f"TIMEOUT after {args.timeout}s"
        finally:
            os.unlink(tmp)
        secs = time.time() - t0
        figs = sorted(f for f in os.listdir(fig_dir) if f.startswith(name + "_") and f.endswith(".png"))
        tail = [l for l in out.strip().splitlines() if l.strip()][-6:]
        err_tail = [l for l in err.strip().splitlines() if l.strip()][-12:] if not ok else []
        summary.append({"notebook": nb, "ok": ok, "seconds": round(secs, 1), "figures": figs,
                        "stdout_tail": tail, "stderr_tail": err_tail, "smoke_patches": sorted(set(applied))})
        print(f"[{'PASS' if ok else 'FAIL'}] {nb:48s} {secs:7.1f}s  {len(figs)} figure(s)", flush=True)
        for l in tail[-3:]:
            print(f"         {l}")
        for l in err_tail:
            print(f"    ERR  {l}")

    with open(os.path.join(fig_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    n_ok = sum(s["ok"] for s in summary)
    print(f"\n{n_ok}/{len(summary)} notebooks ran in {args.mode} mode; figures in {os.path.relpath(fig_dir, ROOT)}/")
    sys.exit(0 if n_ok == len(summary) else 1)


if __name__ == "__main__":
    main()
