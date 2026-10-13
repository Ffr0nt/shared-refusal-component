"""Этап 0 — данные: сплиты по доменам (CPU).

Для каждого домена (источник и колонки — секция dataset конфига; сейчас SALAD-Bench
base_set, домен — 1-category) — непересекающиеся harmful_{train,val,test}.json одного
размера во всех доменах; выборка стратифицирована по stratum_col (лист таксономии,
3-category). Вопросы, встречающиеся в нескольких доменах или повторно,
выброшены. Harmless — общий для всех доменов срез Alpaca, тоже непересекающийся по сплитам;
он копируется в каталог каждого домена (так их читают DIM-пайплайн и rdo.py).

Пишет results/<model>/splits/<domain>/{harmful,harmless}_{train,val,test}.json
и sha256 каждого файла в manifest.json.
"""
import argparse
import os

from shared_refusal.artifacts import Run, write_json
from shared_refusal.config import file_sha256, load_config
from shared_refusal.data import SPLITS, cut, load_harmful, load_harmless, stratified_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run.from_config(cfg)
    sc, dc = cfg["splits"], cfg["dataset"]
    sizes = {s: sc[f"n_{s}"] for s in SPLITS}
    total = sum(sizes.values())

    # harmless: перемешать один раз, нарезать подряд
    harmless = load_harmless(sc["harmless_src"])
    assert len(harmless) >= total, f"harmless {len(harmless)} < {total}"
    harmless = harmless.sample(frac=1, random_state=sc["seed"]).reset_index(drop=True)
    harmless_splits = {s: [{"instruction": q, "source": "alpaca"} for q in part["instruction"]]
                       for s, part in cut(harmless, sizes).items()}

    df = load_harmful(dc, list(cfg["domains"].values()))

    manifest = {"sizes": sizes, "seed": sc["seed"], "harmless_src": sc["harmless_src"],
                "pool": {}, "sha256": {}}
    for d, label in cfg["domains"].items():
        pool = df[df["domain"] == label]
        assert len(pool) >= total, f"{d}: пул {len(pool)} < {total}"
        sample = stratified_sample(pool, total, "stratum", sc["seed"])
        out = run.splits(d)
        for s, part in cut(sample, sizes).items():
            rows = [{"instruction": q, "source": f"{dc['name']}:{d}", "category": c}
                    for q, c in zip(part["text"], part["stratum"])]
            for harmtype, data in (("harmful", rows), ("harmless", harmless_splits[s])):
                p = os.path.join(out, f"{harmtype}_{s}.json")
                write_json(data, p)
                manifest["sha256"][f"{d}/{harmtype}_{s}"] = file_sha256(p)
        manifest["pool"][d] = len(pool)
        print(f"{d}: пул {len(pool)} -> {sizes}, страт {sample['stratum'].nunique()}")

    run.update_manifest("splits", manifest)


if __name__ == "__main__":
    main()
