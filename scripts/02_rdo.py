"""Шаг 02 — одномерные конусы отказа (RDO, Wollschläger et al.) (GPU).

Обучение — rdo.py из форка geometry-of-refusal без изменений метода: одномерный конус
(--seeds), таргеты ablation/addition сгенерированы DIM домена, слой добавления — общая точка
шага 01, фильтр данных включён (как в DIM-пайплайне). Каждый сид задаёт стартовый вектор и
порядок батчей; стартовый вектор сохраняется рядом с результатом.

Один процесс на домен: модель и таргеты грузятся один раз на 5 сидов. Уже обученные сиды
пропускаются, так что прерванный прогон можно просто перезапустить.

Пишет results/<model>/rdo/<domain>/seed_<s>.pt и targets/ (кэш таргетов домена).
"""
import argparse
import os
import subprocess
import sys

from shared_refusal.common import FORK, dim_dir, load_config, rdo_dir, results_dir, splits_dir, update_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen2.5-3b.yaml")
    ap.add_argument("--domains", nargs="+", default=None, help="по умолчанию все из конфига")
    args = ap.parse_args()
    cfg = load_config(args.config)
    domains = args.domains or list(cfg["domains"])

    env = dict(os.environ, SAVE_DIR=results_dir(cfg))
    for d in domains:
        seeds = [str(s) for s in cfg["rdo"]["seeds"][d]]
        cmd = [sys.executable, "rdo.py", "--model", cfg["model"], "--splits", d, "--filter_data",
               "--dim_dir", dim_dir(cfg, d), "--splits_dir", splits_dir(cfg, d),
               "--targets_dir", os.path.join(rdo_dir(cfg, d), "targets"),
               "--out_dir", rdo_dir(cfg, d), "--seeds", *seeds]
        print(f"=== {d}: сиды {' '.join(seeds)} ===", flush=True)
        subprocess.run(cmd, cwd=FORK, env=env, check=True)

    update_manifest(cfg, "rdo", {"seeds": cfg["rdo"]["seeds"], "filter_data": True, "cone_dim": 1})


if __name__ == "__main__":
    main()
