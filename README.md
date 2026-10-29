# shared-refusal-component

Code to reproduce the experiments of the paper (TBD).

We take four harm domains of SALAD-Bench — O1 Representation & Toxicity, O2 Misinformation,
O5 Malicious Use, O6 Human Autonomy — and for each of them compute a difference-in-means
(DIM) refusal direction and train five one-dimensional refusal cones (RDO, Wollschläger et
al.) with distinct seeds. All 24 vectors are intersected with equal weights to obtain one
shared component; we then measure its share of each DIM and test it with directional
ablation — with a first-token proxy on held-out SALAD prompts and with full generations on
StrongREJECT scored by its rubric judge.

## Setup

```bash
git clone https://github.com/Ffr0nt/geometry-of-refusal.git
git clone https://github.com/Ffr0nt/shared-refusal-component.git
cd shared-refusal-component && uv sync
```

Both repositories must be siblings: RDO training and the DIM pipeline come from
`../geometry-of-refusal` (override with `FORK=...`).

## Pipeline

Stages that exchange one artifact — a **direction set** (named unit vectors with
metadata: method, domain, seed, layer). Any stage can be rerun on its own; stages 2 and 4
are CPU-only and cheap, so variants are computed without touching the GPU.

| stage | resource | what it does | output in `results/<run>/` |
|---|---|---|---|
| `00_data` | CPU | disjoint train/val/test (900/128/128) per domain, stratified by leaf category; shared harmless set. With a paired source, each harmful row also carries the dataset's own non-refusing answer (`target`) | `splits/<domain>/` |
| `01_directions` | GPU | DIM per domain at one extraction point (selected once on the pooled val set); one-dimensional RDO cones, 5 seeds per domain; assembled into one set. With `rdo.targets.source: dataset` the RDO targets are written from the data instead of being generated through DIM | `dim/`, `rdo/`, `directions/{directions,init}.pt` |
| `02_component` | CPU | per variant (selection of the set): top eigenvector of `(1/N) Σ v vᵀ`, null band | `component/<variant>/` |
| `03_ablation` | GPU | ASR (first-token proxy) and harmless KL under partial ablation of components, DIM, RDO and random directions | `ablation/` |
| `04_analysis` | CPU | per variant: share of the component in each DIM, cosines, shared-init control | `analysis/<variant>/` |
| `05_convergence` | CPU | how fast the component of N random vectors (DIM and RDO alike) converges: split-half agreement, fit `1/ρ² − 1 = k/N (+ b)`, vectors needed for a target purity | `convergence/` |
| `06_generate` | GPU | StrongREJECT prompts (minus those overlapping our train/val): greedy full answers under ablation of the same directions, plus the first-token proxy on them | `strongreject/{prompts.json,generations/}` |
| `07_judge` | network | StrongREJECT rubric judge (gpt-4o-mini via OpenRouter), cached per answer; score with bootstrap CI, strict subset where the model refuses without ablation, agreement of proxy and refusal phrases with the judge | `strongreject/{judge/,strongreject.json}` |

## Runs

A **run** is one name for three things: `configs/<run>.yaml` (every parameter that affects the
numbers, including the HF revisions of the model and the dataset), `launch/<run>.sh` (that
config pinned by its sha256) and `results/<run>/`.

```bash
GPU=0 bash launch/qwen2.5-3b_salad-o1o2o5o6.sh                     # stages 00–06
STEPS=07 bash launch/qwen2.5-3b_salad-o1o2o5o6.sh                  # judge: needs OPENROUTER_API_KEY
STEPS="02 04 05" bash launch/qwen2.5-3b_salad-o1o2o5o6.sh          # CPU stages only
STEPS=01 bash launch/qwen2.5-3b_salad-o1o2o5o6.sh --assemble-only   # args go to a single stage
GPU=0 bash scripts/tmux.sh launch/qwen2.5-3b_salad-o1o2o5o6.sh     # same in tmux, log in results/<run>/logs/
```

Only `GPU`, `STEPS`, `FORK`, `HF_HOME` (and `OPENROUTER_API_KEY` for `07`) come from the
environment — none of them affects the numbers. `07_judge` needs access to OpenRouter, so it
may run on another machine over a copy of `results/<run>/` (see
[`docs/strongreject.md`](docs/strongreject.md)). The config is fixed in three ways:

- `scripts/pipeline.sh` refuses to start if the config no longer matches the hash in `launch/`;
- every stage records in `manifest.json` the hash of the config sections it depends on, both
  repositories' commits and whether their trees were dirty (`provenance`);
- before starting, a stage checks the recorded hashes of the stages upstream: results of
  different configs never mix in one folder. A new component variant is fine on top of
  finished directions; a changed `rdo` section is not until stage 01 is recomputed.

New run: copy the config under a new name, edit it, write `launch/<new>.sh` with its hash.

`01` skips what is done (DIM if all directions exist, RDO per seed file), so on finished
results it only reassembles the set.

## Layout

- `src/shared_refusal/` — reusable modules; stages talk through `artifacts.DirectionSet`:
  `config`, `artifacts` (run folder, manifest, direction set), `geometry` (spectrum, shared
  component, null level, KL), `data` (stage 0), `models` (per-family chat template, refusal
  tokens, ablation hook — the only source of model-specific knowledge), `directions/`
  (stage 1: `dim`, `rdo` — the only place that calls the fork), `component` (stage 2),
  `ablation` (stage 3), `analysis` (stage 4), `convergence` (stage 5), `strongreject` (stages 6–7: data,
  overlap check, rubric judge, summary), `report`, `provenance` (config hashes per stage,
  commits, HF revision check).
- `scripts/` — general logic: stages 00–07 (thin scripts on top of these modules, `--config` is
  required), `pipeline.sh` (stages in order for one config), `tmux.sh` (a launch file in tmux).
- `configs/`, `launch/` — concrete runs, one pair per run.
- `docs/` — notes on what we do and why ([`docs/NOTES.md`](docs/NOTES.md)), one document per experiment (e.g. [`docs/convergence.md`](docs/convergence.md), judge results on 8 models — [`docs/judge-results.md`](docs/judge-results.md)), figures in `docs/figures/`.
- `tests/` — CPU tests for `src/` and the CPU stages, heavy parts (HF datasets, nnsight model,
  the fork) mocked:

```bash
uv run --with pytest pytest -q
```

A new model family is one entry in `models.FAMILIES` (plus a model class in the fork's DIM
pipeline if the family is new there; `directions.dim.check_family` verifies both agree);
a new dataset is a new `dataset:` section in the config; a new component variant is a new
entry in `component.variants`.

A harmful source comes in one of two shapes, set by `dataset.kind` (see `data.py`):
`labeled` (one domain label per row, no answers — SALAD-Bench) and `paired` (a non-refusing
answer comes with the prompt and labels are multi-label — BeaverTails; domains are assigned by
config order, so rarer categories go first). A paired source makes a control possible:
`rdo.targets.source: dataset` writes the RDO targets from the data, where ordinary targets are
generated by ablating and adding DIM, so RDO no longer learns from DIM at all
(`directions/targets.py`). It requires `rdo.hparams.addition_lambda: 0`, since the refusal
target of the addition term has no counterpart in the data. The run
`qwen2.5-3b_beavertails-b1b5b6bv` is that control; it is written up in
`concept_dim/experiments/natural_targets/`.

Every step records what it produced (extraction point, seeds, split hashes, headline numbers)
in `results/<run>/manifest.json`; markdown tables go to `results/<run>/reports/`.
