"""Этап 1 — направления отказа (GPU): DIM каждого домена, одномерные конусы RDO, единый набор.

1. DIM (пайплайн Arditi et al. из форка):
   - фильтр: harmful оставляем, если модель на нём отказывает, harmless — если не отказывает;
   - mean_diffs [n_pos, n_layer, hidden] домена: harmful_train домена против ОБЩЕГО
     отфильтрованного harmless_train — DIM доменов различаются только harmful-стороной;
   - точка (pos, layer) выбирается ОДИН раз: select_direction на объединённом harmful_val по
     кандидатам объединённой выборки (среднее mean_diffs доменов);
   - DIM домена = его mean_diffs в этой точке.
2. RDO (rdo.py форка): одномерный конус на каждый сид, таргеты — по DIM домена, слой
   добавления — точка из п. 1; шаблон, refusal-токены и гиперпараметры — явно.
3. Единый набор directions/directions.pt (+ стартовые векторы RDO в directions/init.pt).

Готовое не пересчитывается: DIM — если есть direction.pt всех доменов и точка в manifest,
RDO — по файлам сидов. Поэтому прерванный прогон можно просто перезапустить; на готовых
результатах этап только пересобирает набор (без GPU).
"""
import argparse
import os

import torch

from shared_refusal import models
from shared_refusal.artifacts import Run, read_json, write_json
from shared_refusal.config import load_config
from shared_refusal.directions import assemble, dim, rdo


def instructions(run, domain, harmtype, split):
    return [r["instruction"] for r in read_json(os.path.join(run.splits(domain), f"{harmtype}_{split}.json"))]


def dim_done(run, domains):
    return "dim" in run.manifest() and all(
        os.path.exists(os.path.join(run.dim(d), "direction.pt")) for d in domains)


def compute_dim(cfg, run, fam):
    domains = list(cfg["domains"])
    mb = dim.model_base(cfg["model"])
    dim.check_family(mb, fam)

    first = domains[0]
    harmless_train = dim.filter_by_refusal(mb, instructions(run, first, "harmless", "train"), refuse=False)
    harmless_val = dim.filter_by_refusal(mb, instructions(run, first, "harmless", "val"), refuse=False)
    counts = {"harmless_train": len(harmless_train), "harmless_val": len(harmless_val)}

    mean_diffs, harmful_val = {}, []
    for d in domains:
        md_path = os.path.join(run.dim(d), "generate_directions", "mean_diffs.pt")
        if os.path.exists(md_path):
            mean_diffs[d] = torch.load(md_path, map_location="cpu", weights_only=True)
            print(f"{d}: mean_diffs уже есть — {md_path}")
        else:
            harmful_train = dim.filter_by_refusal(mb, instructions(run, d, "harmful", "train"), refuse=True)
            counts[f"{d}/harmful_train"] = len(harmful_train)
            mean_diffs[d] = dim.mean_diffs(mb, harmful_train, harmless_train, os.path.dirname(md_path))
        hv = dim.filter_by_refusal(mb, instructions(run, d, "harmful", "val"), refuse=True)
        counts[f"{d}/harmful_val"] = len(hv)
        harmful_val += hv
        print(f"{d}: harmful_val после фильтра {len(hv)}")

    union = dim.union(mean_diffs)
    os.makedirs(os.path.join(run.dim("union"), "generate_directions"), exist_ok=True)
    torch.save(union, os.path.join(run.dim("union"), "generate_directions", "mean_diffs.pt"))
    pos, layer = dim.select_point(mb, harmful_val, harmless_val, union, run.path("dim", "select"))
    print(f"общая точка: pos={pos} layer={layer}")

    norms = {}
    for d, md in list(mean_diffs.items()) + [("union", union)]:
        v = dim.at_point(md, pos, layer)
        torch.save(v, os.path.join(run.dim(d), "direction.pt"))
        write_json({"pos": pos, "layer": layer}, os.path.join(run.dim(d), "direction_metadata.json"))
        norms[d] = float(v.norm())
    run.update_manifest("dim", {"pos": pos, "layer": layer, "norms": norms, "counts": counts})
    del mb
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domains", nargs="+", default=None,
                    help="обучить RDO только для этих доменов (на результат не влияет — для разбиения работы)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    fam = models.family(cfg["model"])
    domains = list(cfg["domains"])
    rc = cfg["rdo"]

    if dim_done(run, domains):
        print("DIM: готово, пропуск")
    else:
        compute_dim(cfg, run, fam)

    for d in args.domains or domains:
        todo = rdo.pending(run.rdo(d), rc["seeds"][d])
        if not todo:
            print(f"RDO {d}: все сиды готовы, пропуск")
            continue
        print(f"=== RDO {d}: сиды {' '.join(map(str, todo))} ===", flush=True)
        cmd = rdo.command(cfg["model"], fam, d, run.dim(d), run.splits(d), os.path.join(run.rdo(d), "targets"),
                          run.rdo(d), todo, rc["hparams"])
        rdo.train(cmd, save_dir=run.root)
    run.update_manifest("rdo", {"seeds": rc["seeds"], "hparams": rc["hparams"], "cone_dim": 1})

    missing = {d: rdo.pending(run.rdo(d), rc["seeds"][d]) for d in domains}
    if any(missing.values()):
        print(f"набор не собран — не хватает сидов: { {d: s for d, s in missing.items() if s} }")
        return
    point = {k: run.manifest()["dim"][k] for k in ("pos", "layer")}
    dims = {d: torch.load(os.path.join(run.dim(d), "direction.pt"), map_location="cpu", weights_only=True)
            for d in domains}
    rdos = {d: {s: rdo.load_seed(run.rdo(d), s) for s in rc["seeds"][d]} for d in domains}
    ds, init = assemble(dims, rdos, point)
    ds.save(run.path("directions", "directions.pt"))
    init.save(run.path("directions", "init.pt"))
    run.update_manifest("directions", {"n": len(ds), "names": ds.names, "point": point})
    print(f"набор: {len(ds)} направлений -> {run.path('directions')}")


if __name__ == "__main__":
    main()
