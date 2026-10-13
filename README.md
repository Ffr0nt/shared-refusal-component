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

| step | resource | what it does | output in `results/<model>/` |
|---|---|---|---|
| `00` | CPU | disjoint train/val/test (900/128/128) per domain, stratified by leaf category; shared harmless set | `splits/<domain>/` |
| `01` | GPU | DIM per domain at one extraction point (pos, layer), selected once on the pooled val set | `dim/<domain>/`, `dim/select/` |
| `02` | GPU | one-dimensional RDO cones, seeds 1–20 (5 per domain); init vector saved | `rdo/<domain>/seed_XX.pt` |
| `03` | CPU | shared component: top eigenvector of `(1/24) Σ v vᵀ`, null band | `component/` |
| `04` | CPU | share of the component in each DIM; shared-init control | `energy/` |
| `05` | GPU | ASR (first-token proxy) under partial ablation vs. harmless KL | `ablation/` |

```bash
bash run.sh 00
CUDA_VISIBLE_DEVICES=0 bash run.sh 01
CUDA_VISIBLE_DEVICES=0 bash run.sh 02            # or: bash run.sh 02 --domains O1 O2
bash run.sh 03
bash run.sh 04
CUDA_VISIBLE_DEVICES=0 bash run.sh 05
```

## Layout

- `src/shared_refusal/` — reusable modules, independent of the model and the dataset:
  `common` (config, paths, manifest, reports, vector loading), `geometry` (spectrum, shared
  component, null level, KL), `data` (harmful source from the `dataset:` config section,
  harmless, splits), `models` (per-family chat template, refusal tokens, ablation hook).
- `scripts/` — pipeline steps 00–05 on top of these modules.
- `docs/` — notes on what we do and why ([`docs/NOTES.md`](docs/NOTES.md)).
- `tests/` — CPU tests for `src/` with heavy parts (HF datasets, nnsight model) mocked:

```bash
uv run --with pytest pytest -q
```

A new model family is one entry in `models.FAMILIES`; a new dataset is a new `dataset:`
section in the config.

Every step records what it produced (extraction point, seeds, split hashes, headline numbers)
in `results/<model>/manifest.json`; markdown tables go to `results/<model>/reports/`.
Configuration: [`configs/qwen2.5-3b.yaml`](configs/qwen2.5-3b.yaml).
