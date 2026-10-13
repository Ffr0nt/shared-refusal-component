"""Шаг 05 — аблация: снимает ли общая компонента отказ (GPU).

Интервенция — частичная аблация направления во всех слоях (вход блока, выход attn, выход
mlp — как в rdo.py): h <- h - a (h·v) v, a ∈ alphas. Коэффициент a — ручка для KL:
сравнивать направления честно только при близком KL.

Метрики (прокси по первому токену, judge-based ASR — позже):
  ASR — доля harmful_test домена с refusal_metric < 0 (отказ снят), по каждому домену;
  KL  — на harmless_test между распределениями следующего токена до и после аблации,
        kl_div_fn(baseline, ablated) с reduction='batchmean', как в rdo.py.

Направления: общая компонента w, DIM каждого домена, каждый из 20 векторов RDO, n_rand
случайных. Test-сплиты не фильтруются: базовый ASR без аблации печатается отдельно.

Пишет results/<model>/ablation/ablation.json и reports/ablation.md.
"""
import argparse
import os

import torch

from shared_refusal import models
from shared_refusal.common import load_config, load_vectors, read_json, results_dir, splits_dir, write_json, write_report
from shared_refusal.geometry import kl, random_directions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ac = cfg["ablation"]
    domains = list(cfg["domains"])

    from scoring import refusal_metric

    fam = models.family(cfg["model"])
    template, refusal_toks = fam["template"], fam["refusal_toks"]
    model = models.load(cfg["model"])

    def prompts(domain, harmtype):
        rows = read_json(os.path.join(splits_dir(cfg, domain), f"{harmtype}_test.json"))
        return [template.format(instruction=r["instruction"]) for r in rows]

    harmful = {d: prompts(d, "harmful") for d in domains}
    harmless = prompts(domains[0], "harmless")

    names, V = load_vectors(cfg)
    w = torch.load(results_dir(cfg, "component", "component.pt"), map_location="cpu", weights_only=True)["w"]
    dirs = [("w", w.float())] + [(n, v.float()) for n, v in zip(names, V)]
    dirs += [(f"rand_{i}", r) for i, r in enumerate(random_directions(w.shape[0], ac["n_rand"], seed=0))]

    bs = ac["batch_size"]

    def run(ps, v, a, want):
        v = None if v is None else (v / v.norm()).to(model.dtype).to(model.device)
        out = []
        for i in range(0, len(ps), bs):
            with model.trace(ps[i:i + bs]):
                if v is not None:
                    models.ablate(model, v, a)
                logits = model.lm_head.output[:, -1]
                s = (refusal_metric(logits, refusal_toks) if want == "metric" else logits).save()
            out.append(s.value.detach().float().cpu())
            torch.cuda.empty_cache()
        return torch.cat(out)

    base_asr = {d: float((run(harmful[d], None, 0, "metric") < 0).float().mean()) for d in domains}
    base_logits = run(harmless, None, 0, "logits")
    print("без аблации: " + "  ".join(f"{d}={x:.3f}" for d, x in base_asr.items()), flush=True)

    rows = []
    for name, v in dirs:
        for a in ac["alphas"]:
            asr = {d: float((run(harmful[d], v, a, "metric") < 0).float().mean()) for d in domains}
            row = {"direction": name, "alpha": a, "kl": kl(base_logits, run(harmless, v, a, "logits")),
                   "asr": asr, "asr_mean": sum(asr.values()) / len(asr)}
            rows.append(row)
            print(f"{name:14s} a={a:.2f}  KL={row['kl']:.4f}  ASR=" +
                  " ".join(f"{d}:{x:.3f}" for d, x in asr.items()), flush=True)

    write_json({"base_asr": base_asr, "n_test": {d: len(p) for d, p in harmful.items()},
                "n_harmless": len(harmless), "rows": rows}, results_dir(cfg, "ablation", "ablation.json"))

    # отчёт: w, DIM, случайные — построчно; RDO — среднее по 5 сидам домена
    def agg(label, rs):
        return {"direction": label, "kl": sum(r["kl"] for r in rs) / len(rs),
                "asr": {d: sum(r["asr"][d] for r in rs) / len(rs) for d in domains},
                "asr_mean": sum(r["asr_mean"] for r in rs) / len(rs)}

    table = []
    for a in ac["alphas"]:
        at = [r for r in rows if r["alpha"] == a]
        table += [dict(r) for r in at if not r["direction"].startswith(("rdo_", "rand_"))]
        table += [agg(f"rdo_{d} (5 сидов)", [r for r in at if r["direction"].startswith(f"rdo_{d}_")]) for d in domains]
        table += [agg("rand (среднее)", [r for r in at if r["direction"].startswith("rand_")])]
        for r in table[-(len(domains) + 1):]:
            r["alpha"] = a
    lines = [f"# Аблация — {cfg['model_alias']}", "",
             "ASR — доля harmful_test с refusal_metric < 0; KL — на harmless_test. "
             "**Сравнивать направления только при близком KL.**", "",
             "Без аблации: " + ", ".join(f"{d} {x:.3f}" for d, x in base_asr.items()) + ".", "",
             "| направление | a | KL | " + " | ".join(f"ASR {d}" for d in domains) + " | ASR среднее |",
             "|---|---|---|" + "---|" * (len(domains) + 1)]
    lines += [f"| {r['direction']} | {r['alpha']:.2f} | {r['kl']:.4f} | " +
              " | ".join(f"{r['asr'][d]:.3f}" for d in domains) + f" | **{r['asr_mean']:.3f}** |" for r in table]
    write_report(cfg, "ablation.md", lines)


if __name__ == "__main__":
    main()
