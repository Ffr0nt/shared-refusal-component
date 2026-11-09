"""Этап 0 — данные: сплиты по доменам (CPU).

Для каждого домена (источник, вид и колонки — секция dataset конфига: SALAD-Bench с одной
меткой домена или BeaverTails с готовыми ответами, см. data.py) — непересекающиеся
harmful_{train,val,test}.json одного размера во всех доменах; выборка стратифицирована по
страте (лист таксономии у SALAD, категория внутри домена у BeaverTails). Harmless — общий для
всех доменов срез Alpaca, тоже непересекающийся по сплитам; он копируется в каталог каждого
домена (так их читают DIM-пайплайн и rdo.py).

Если у источника есть готовые ответы (dataset.kind: paired), ответ запроса пишется в поле
target: из него этап 1 делает цель обучения RDO, в которой нет DIM.

Пишет results/<model>/splits/<domain>/{harmful,harmless}_{train,val,test}.json
и sha256 каждого файла в manifest.json.
"""
import argparse
import os

from shared_refusal import provenance
from shared_refusal.artifacts import Run, write_json
from shared_refusal.config import file_sha256, load_config
from shared_refusal.data import SPLITS, cut, load_harmful, load_harmless, stratified_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    provenance.check_upstream(run, cfg, "data")
    sc, dc = cfg["splits"], cfg["dataset"]
    sizes = {s: sc[f"n_{s}"] for s in SPLITS}
    total = sum(sizes.values())

    # harmless: перемешать один раз, нарезать подряд
    harmless = load_harmless(sc["harmless_src"])
    assert len(harmless) >= total, f"harmless {len(harmless)} < {total}"
    harmless = harmless.sample(frac=1, random_state=sc["seed"]).reset_index(drop=True)
    harmless_splits = {s: [{"instruction": q, "source": "alpaca"} for q in part["instruction"]]
                       for s, part in cut(harmless, sizes).items()}

    df = load_harmful(dc, cfg["domains"])
    has_target = "target" in df.columns

    manifest = {"sizes": sizes, "seed": sc["seed"], "harmless_src": sc["harmless_src"],
                "pool": {}, "sha256": {}}
    for d in cfg["domains"]:
        pool = df[df["domain"] == d]
        assert len(pool) >= total, f"{d}: пул {len(pool)} < {total}"
        sample = stratified_sample(pool, total, "stratum", sc["seed"])
        out = run.splits(d)
        for s, part in cut(sample, sizes).items():
            rows = [{"instruction": r.text, "source": f"{dc['name']}:{d}", "category": r.stratum,
                     **({"target": r.target} if has_target else {})}
                    for r in part.itertuples(index=False)]
            for harmtype, data in (("harmful", rows), ("harmless", harmless_splits[s])):
                p = os.path.join(out, f"{harmtype}_{s}.json")
                write_json(data, p)
                manifest["sha256"][f"{d}/{harmtype}_{s}"] = file_sha256(p)
        manifest["pool"][d] = len(pool)
        print(f"{d}: пул {len(pool)} -> {sizes}, страт {sample['stratum'].nunique()}")

    run.update_manifest("splits", manifest)
    provenance.record(run, cfg, "data")


if __name__ == "__main__":
    main()
