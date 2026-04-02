# LUMI-G Container Setup — Session Summary

## Goal

Build a single Singularity container (`graph-found-ddsbm-directo-<date>.sif`) on
LUMI-G (`standard-g` partition) that supports three codebases simultaneously:

| Repo | Purpose |
|---|---|
| Private GitLab (GPS) | Graph GPS multi-GPU training on ZINC |
| `ArslanArdavic/DDSBM` | Discrete Diffusion Schrödinger Bridge Matching for graph transformation |
| `ArslanArdavic/DIRECTO` | Directed graph generation with dual attention — architecture to integrate into DDSBM |

---

## Starting Point

- Base image: `lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif`
- Already built: `graph-found-20260326.sif` — extends the base with `torch_geometric` and
  `scikit-learn`, following the procedure in `LUMInewcontainers.pdf`
- Python **3.12**, PyTorch **2.9.1+ROCm6.4**, torch_geometric **2.7.0**

---

## Why Containers on LUMI

Two reasons from LUMI documentation:

1. **ROCm / Slingshot compatibility** — containers carry the exact ROCm and
   libfabric versions that match the hardware
2. **Filesystem friendliness** — a pip install creates thousands of small files
   which stress Lustre and can exhaust inode quotas. Everything baked into a `.sif`
   appears to Lustre as **one file**

---

## Dependency Reconciliation

### Version conflicts between codebases and base image

| Package | Base image | DDSBM requires | DIRECTO requires | Resolution |
|---|---|---|---|---|
| `torch` | 2.9.1+rocm6.4 | 2.4.0 | 2.4.0 | Accept higher — do **not** downgrade (would lose ROCm 6.4) |
| `torch_geometric` | 2.7.0 | ==2.3.1 | ==2.3.1 | Accept higher — APIs used are stable |
| `pytorch-lightning` | 2.6.1 | ==2.0.4 | ==2.0.4 | Accept higher — patch DDSBM for PL 2.6.x API |
| `torchmetrics` | 1.8.2 | ==0.11.4 | ==0.11.4 | Accept higher — patch DDSBM for 1.x API |
| `numpy` | 2.2.6 | <2.0 | any | No action — `np.Inf` bug only affects PL 2.0.4 which we are not using |
| `wandb` | 0.25.0 | ==0.20.1 | ==0.20.1 | Accept higher |
| `dill` | 0.4.0 | ==0.3.9 | — | Accept higher — API compatible |
| `setuptools` | 79.0.1 | ==68.0.0 | ==68.0.0 | **Do not downgrade** — would corrupt pip metadata |

### Packages absent from base image that must be added

| Package | Required by |
|---|---|
| `omegaconf==2.3.0` | DDSBM + DIRECTO |
| `hydra-core==1.3.2` | DDSBM + DIRECTO |
| `rdkit` | DDSBM + DIRECTO |
| `PyGSP==0.5.1` | DDSBM + DIRECTO |
| `imageio==2.31.1` | DDSBM + DIRECTO |
| `seaborn==0.13.2` | DDSBM + DIRECTO |
| `overrides==7.3.1` | DDSBM + DIRECTO |
| `gpustat==0.6.0` | DDSBM + DIRECTO |
| `graph-tool==2.97` | DDSBM + DIRECTO (apt, not pip) |
| `pygmtools==0.5.3` | DDSBM only |
| `toolz` | DDSBM only |
| `pyemd==1.0.0` | DDSBM only (spectre_utils.py) |
| `fire==0.7.0` | DDSBM only (analysis scripts) |
| `fcd-torch==1.0.7` | DDSBM only (experiments/analysis/fcd.py) |
| `EDeN` (GitHub) | DDSBM only (NSPDK evaluation) |

### Packages deliberately omitted

| Package | Reason |
|---|---|
| `dgl` | No ROCm wheel exists; not imported in DIRECTO model files being integrated |
| `psi4` | Quantum chemistry engine; not used in graph generation code paths |
| `torchdata` | pixi.toml only; no `import torchdata` found in any `src/` file |
| `black` | Formatter; not a runtime dependency |
| `setuptools==68.0.0` | Do not downgrade from base 79.0.1 |

---

## `pip install -e .` — Decision

Both DDSBM and DIRECTO have a `pyproject.toml` with **no `dependencies` field**.
`pip install -e .` installs zero packages. Its only effects are:

1. Registers `import ddsbm` / `import directo` by writing a pointer into site-packages
2. Creates `ddsbm-train` / `ddsbm-test` shell scripts (DDSBM only)
3. Anchors `PROJECT_ROOT` via `__file__` for path resolution (DDSBM only)

**Decision: do not use `pip install -e .`**. Use `PYTHONPATH` instead — it is
faster, writes nothing to Lustre, and is consistent with the existing GPS SLURM
script pattern:

```bash
export PYTHONPATH=/scratch/project_465002822/DDSBM/src:/scratch/project_465002822/DIRECTO/src
```

Call `python src/main.py` directly instead of the `ddsbm-train` wrapper.

---

## Bugs Found and Fixed in `.def`

| # | Bug | Fix |
|---|---|---|
| 1 | `. /opt/venv/bin/activate` missing — pip targets system Python | Added as first line of `%post` |
| 2 | `--no-deps` on hydra-core skips real transitive dependencies | Removed; install omegaconf first, then hydra-core separately |
| 3 | `dill==0.3.9` would downgrade base image's 0.4.0 | Removed pin; 0.4.0 is compatible |
| 4 | `SITE=$(python3 ...)` resolves system Python path, not venv | Changed to `$(/opt/venv/bin/python ...)` |

---

## Bugs Found in Source Code (fix in repos, not in `.def`)

### DDSBM
- **`torchmetrics` 1.x API**: metric classes moved to submodules
  (`from torchmetrics.regression import MeanAbsoluteError`). Patch all metric
  imports in `src/ddsbm/`.
- **`pytorch-lightning` 2.6.x**: add explicit `on_step`/`on_epoch`/`sync_dist`
  flags to all `self.log()` calls.

### DIRECTO — `setup_problems.md` Problems 1–7
All seven problems documented. The one that requires a source code patch:

**Problem 6** — `self.spe_out_dim` used before assignment in
`src/models/transformer_model_directed.py`. Uncomment:
```python
self.spe_out_dim = spe_dims["out_dim"] if pos_enc == "spe" else 0
```

---

## Orca Binary (DDSBM)

`src/ddsbm/analysis/orca/orca.cpp` must be compiled once manually after the
container is built. Only needed for unconditional generation on comm20/planar
datasets — **not needed for zinc**.

```bash
srun --account=project_465002822 --partition=small \
     --nodes=1 --ntasks=1 --time=00:05:00 \
     singularity run -B /scratch/project_465002822 $SIF \
         g++ -O2 -std=c++11 \
             -o /scratch/project_465002822/DDSBM/src/ddsbm/analysis/orca/orca \
                /scratch/project_465002822/DDSBM/src/ddsbm/analysis/orca/orca.cpp
```

---

## Three Actions Required Outside the Container

1. **Patch DIRECTO** — uncomment `self.spe_out_dim` in `transformer_model_directed.py`
2. **Set `PYTHONPATH`** in all SLURM scripts:
   ```bash
   export PYTHONPATH=/scratch/project_465002822/DDSBM/src:/scratch/project_465002822/DIRECTO/src
   ```
3. **Confirm DGL not used** in DIRECTO components being integrated:
   ```bash
   grep -r "import dgl\|from dgl" /scratch/project_465002822/DIRECTO/src/models/
   ```

---

## Build Procedure

```bash
cd /project/project_465002822/containers

# 1. Start an interactive job with enough RAM for singularity build
srun --account=project_465002822 --partition=small \
     --time=01:00:00 --nodes=1 --mem=128G --cpus-per-task=8 --pty bash

# 2. Load build modules
module load CrayEnv PRoot

# 3. Build
singularity build graph-found-ddsbm-directo-$(date +%Y%m%d).sif \
    ./defs/graph-found-ddsbm-directo-$(date +%Y%m%d).def
```

## Validation

```bash
export SIF=/project/project_465002822/containers/graph-found-ddsbm-directo-<date>.sif

srun --account=project_465002822 --partition=dev-g \
     --nodes=1 --ntasks=1 --gpus-per-task=1 \
     --cpus-per-task=7 --mem-per-gpu=60G --time=00:10:00 \
     singularity run $SIF python -c "
import torch, torch_geometric, pytorch_lightning as pl
import torchmetrics, hydra, graph_tool, rdkit, pygmtools
print('torch:       ', torch.__version__)
print('pyg:         ', torch_geometric.__version__)
print('lightning:   ', pl.__version__)
print('torchmetrics:', torchmetrics.__version__)
print('GPUs:        ', torch.cuda.device_count())
"
```
