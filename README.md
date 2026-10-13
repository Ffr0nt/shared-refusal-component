# shared-refusal-component

Code to reproduce the experiments of the paper (TBD).

We take four harm domains of SALAD-Bench — O1 Representation & Toxicity, O2 Misinformation,
O5 Malicious Use, O6 Human Autonomy — and for each of them compute a difference-in-means
(DIM) refusal direction and train five one-dimensional refusal cones (RDO, Wollschläger et
al.) with distinct seeds. All 24 vectors are intersected with equal weights to obtain one
shared component; we then measure its share of each DIM and test it with directional
ablation.

## Setup

```bash
git clone https://github.com/Ffr0nt/geometry-of-refusal.git
git clone https://github.com/Ffr0nt/shared-refusal-component.git
cd shared-refusal-component && uv sync
```

Both repositories must be siblings: RDO training and the DIM pipeline come from
`../geometry-of-refusal` (override with `FORK=...`).

## Pipeline

Four stages that exchange one artifact — a **direction set** (named unit vectors with
metadata: method, domain, seed, layer). Any stage can be rerun on its own; stages 2 and 4
are CPU-only and cheap, so variants are computed without touching the GPU.

| stage | resource | what it does | output in `results/<model>/` |
|---|---|---|---|
| `00_data` | CPU | disjoint train/val/test (900/128/128) per domain, stratified by leaf category; shared harmless set | `splits/<domain>/` |
| `01_directions` | GPU | DIM per domain at one extraction point (selected once on the pooled val set); one-dimensional RDO cones, 5 seeds per domain; assembled into one set | `dim/`, `rdo/`, `directions/{directions,init}.pt` |
| `02_component` | CPU | per variant (selection of the set): top eigenvector of `(1/N) Σ v vᵀ`, null band | `component/<variant>/` |
| `03_ablation` | GPU | ASR (first-token proxy) and harmless KL under partial ablation of components, DIM, RDO and random directions | `ablation/` |
| `04_analysis` | CPU | per variant: share of the component in each DIM, cosines, shared-init control | `analysis/<variant>/` |

```bash
CONFIG=configs/qwen2.5-3b.yaml
bash run.sh 00
CUDA_VISIBLE_DEVICES=0 bash run.sh 01            # or: … run.sh 01 --domains O1 O2 (RDO only for these)
bash run.sh 02                                   # or: bash run.sh 02 --variants all
CUDA_VISIBLE_DEVICES=0 bash run.sh 03
bash run.sh 04
```

`01` skips what is done (DIM if all directions exist, RDO per seed file), so on finished
results it only reassembles the set.

## Layout

- `src/shared_refusal/` — reusable modules; stages talk through `artifacts.DirectionSet`:
  `config`, `artifacts` (run folder, manifest, direction set), `geometry` (spectrum, shared
  component, null level, KL), `data` (stage 0), `models` (per-family chat template, refusal
  tokens, ablation hook — the only source of model-specific knowledge), `directions/`
  (stage 1: `dim`, `rdo` — the only place that calls the fork), `component` (stage 2),
  `ablation` (stage 3), `analysis` (stage 4), `report`.
- `scripts/` — stages 00–04, thin scripts on top of these modules; `--config` is required.
- `docs/` — notes on what we do and why ([`docs/NOTES.md`](docs/NOTES.md)).
- `tests/` — CPU tests for `src/` and the CPU stages, heavy parts (HF datasets, nnsight model,
  the fork) mocked:

```bash
uv run --with pytest pytest -q
```

A new model family is one entry in `models.FAMILIES` (plus a model class in the fork's DIM
pipeline if the family is new there; `directions.dim.check_family` verifies both agree);
a new dataset is a new `dataset:` section in the config; a new component variant is a new
entry in `component.variants`.

Every step records what it produced (extraction point, seeds, split hashes, headline numbers)
in `results/<model>/manifest.json`; markdown tables go to `results/<model>/reports/`.
Configuration: [`configs/qwen2.5-3b.yaml`](configs/qwen2.5-3b.yaml).
