"""Шаг 01 — DIM каждого домена в одной общей точке (pos, layer) (GPU).

1. Фильтр, как в пайплайне Arditi et al.: harmful оставляем, если модель на нём отказывает
   (refusal score > 0), harmless — если не отказывает (< 0).
2. mean_diffs [n_pos, n_layer, hidden] каждого домена: harmful_train домена против ОБЩЕГО
   отфильтрованного harmless_train — DIM доменов различаются только harmful-стороной.
3. Точка выбирается ОДИН раз: select_direction на объединённом harmful_val всех доменов по
   кандидатам-DIM объединённой выборки (среднее mean_diffs доменов = равные веса доменов).
4. DIM домена = его mean_diffs в этой точке.

Пишет results/<model>/dim/<domain>/{direction.pt, direction_metadata.json,
generate_directions/mean_diffs.pt} (раскладка, которую читает rdo.py), dim/union/,
dim/select/ (оценки всех точек) и точку в manifest.json.
"""
import argparse
import os
import sys

import torch

from shared_refusal.common import FORK, dim_dir, load_config, read_json, results_dir, splits_dir, update_manifest, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    sys.path.insert(0, os.path.join(FORK, "refusal_direction"))
    from pipeline.model_utils.model_factory import construct_model_base
    from pipeline.submodules.generate_directions import generate_directions
    from pipeline.submodules.select_direction import get_refusal_scores, select_direction

    model_base = construct_model_base(cfg["model"])

    def instructions(domain, harmtype, split):
        return [r["instruction"] for r in read_json(os.path.join(splits_dir(cfg, domain), f"{harmtype}_{split}.json"))]

    def keep(xs, refuse):
        scores = get_refusal_scores(model_base.model, xs, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        return [x for x, s in zip(xs, scores.tolist()) if (s > 0) == refuse]

    first = next(iter(cfg["domains"]))
    harmless_train = keep(instructions(first, "harmless", "train"), refuse=False)
    harmless_val = keep(instructions(first, "harmless", "val"), refuse=False)
    counts = {"harmless_train": len(harmless_train), "harmless_val": len(harmless_val)}

    mean_diffs, harmful_val = {}, []
    for d in cfg["domains"]:
        md_path = os.path.join(dim_dir(cfg, d), "generate_directions", "mean_diffs.pt")
        if os.path.exists(md_path):
            mean_diffs[d] = torch.load(md_path, map_location="cpu", weights_only=True)
            print(f"{d}: mean_diffs уже есть — {md_path}")
        else:
            harmful_train = keep(instructions(d, "harmful", "train"), refuse=True)
            counts[f"{d}/harmful_train"] = len(harmful_train)
            mean_diffs[d] = generate_directions(model_base, harmful_train, harmless_train,
                                                artifact_dir=os.path.dirname(md_path)).cpu()
        hv = keep(instructions(d, "harmful", "val"), refuse=True)
        counts[f"{d}/harmful_val"] = len(hv)
        harmful_val += hv
        print(f"{d}: harmful_val после фильтра {len(hv)}")

    union = torch.stack(list(mean_diffs.values())).mean(0)
    os.makedirs(os.path.join(dim_dir(cfg, "union"), "generate_directions"), exist_ok=True)
    torch.save(union, os.path.join(dim_dir(cfg, "union"), "generate_directions", "mean_diffs.pt"))

    pos, layer, _ = select_direction(model_base, harmful_val, harmless_val, union.to(model_base.model.device),
                                     artifact_dir=results_dir(cfg, "dim", "select"))
    pos, layer = int(pos), int(layer)
    print(f"общая точка: pos={pos} layer={layer}")

    norms = {}
    for d, md in list(mean_diffs.items()) + [("union", union)]:
        v = md[pos, layer]
        torch.save(v, os.path.join(dim_dir(cfg, d), "direction.pt"))
        write_json({"pos": pos, "layer": layer}, os.path.join(dim_dir(cfg, d), "direction_metadata.json"))
        norms[d] = float(v.norm())

    update_manifest(cfg, "dim", {"pos": pos, "layer": layer, "norms": norms, "counts": counts})


if __name__ == "__main__":
    main()
