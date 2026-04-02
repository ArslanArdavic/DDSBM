# PATCHING.md

Required source code patches for running DDSBM and DIRECTO on LUMI-G with the
`graph-found-ddsbm-directo.sif` container. The container ships
`pytorch-lightning==2.6.1` and `torchmetrics==1.8.2` — both higher than what the
codebases were written against. These patches bring the code into alignment with
those versions without touching the container.

---

## DIRECTO

### Patch 1 — `self.spe_out_dim` used before assignment

**File:** `src/models/transformer_model_directed.py`

**Symptom:** `AttributeError: 'GraphTransformerDirected' object has no attribute
'spe_out_dim'` on startup, regardless of which positional encoding is selected.

**Cause:** The line was commented out during a refactor but is still referenced two
lines later in the `mlp_in_X` and `mlp_in_E` layer definitions.

**Fix:** Uncomment the line in `__init__`:

```python
self.spe_out_dim = spe_dims["out_dim"] if pos_enc == "spe" else 0
```

---

## DDSBM

### Patch 2 — `torchmetrics` 1.x API

**Symptom:** `ImportError` or `AttributeError` on any metric class imported
directly from the `torchmetrics` top-level namespace.

**Cause:** `torchmetrics==0.11.4` exposed metric classes at the top level.
`torchmetrics==1.x` moved them into submodules. The classes and their APIs are
otherwise identical.

**Fix:** Update all metric imports in `src/ddsbm/`. Find affected lines with:

```bash
grep -rn "from torchmetrics import\|torchmetrics\." src/ddsbm/
```

Apply the following renames:

| Old (0.11.x) | New (1.x) |
|---|---|
| `from torchmetrics import MeanAbsoluteError` | `from torchmetrics.regression import MeanAbsoluteError` |
| `from torchmetrics import MeanSquaredError` | `from torchmetrics.regression import MeanSquaredError` |
| `from torchmetrics import MetricCollection` | `from torchmetrics import MetricCollection` (unchanged) |

---

### Patch 3 — `pytorch-lightning` 2.6.x `self.log()` defaults

**Symptom:** `MisconfigurationException` or silent metric aggregation errors
during training or validation.

**Cause:** `pytorch-lightning==2.0.4` had permissive defaults for `on_step` and
`on_epoch` inside `training_step` and `validation_step`. Version 2.5+ tightened
these defaults and emits errors when the combination is ambiguous.

**Fix:** Add explicit flags to every `self.log()` call in `src/ddsbm/`. Find
affected lines with:

```bash
grep -rn "self\.log(" src/ddsbm/
```

Apply the following pattern:

```python
# Before
self.log("train/loss", loss)

# After
self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=True)
```

Use `on_step=False, on_epoch=True` for validation and test metrics:

```python
self.log("val/NLL", nll, on_step=False, on_epoch=True, sync_dist=True)
```

---

## Hydra config patch (DIRECTO)

### Patch 4 — `cfg.general.conditional` key missing

**File:** `configs/general/general_default.yaml`

**Symptom:** `omegaconf.errors.ConfigAttributeError: Key 'conditional' not in
struct` on startup.

**Cause:** The model initialization code reads `cfg.general.conditional` but the
key was never added to the default config.

**Fix:** Add to `configs/general/general_default.yaml`:

```yaml
conditional: False
```

---

## Verification

After applying all patches, run the DIRECTO debug mode and DDSBM graph matching
step to confirm clean startup:

```bash
# DIRECTO
singularity run -B /scratch/project_465002822 $SIF \
    bash -c 'cd /scratch/project_465002822/DIRECTO/src && \
             PYTHONPATH=/scratch/project_465002822/DIRECTO/src \
             python main.py +experiment=debug'

# DDSBM
singularity run -B /scratch/project_465002822 $SIF \
    bash -c 'PYTHONPATH=/scratch/project_465002822/DDSBM/src \
             python /scratch/project_465002822/DDSBM/src/main.py \
             dataset.name=zinc general.name=test_run general.gpus=1'
```
