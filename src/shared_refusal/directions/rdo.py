"""Одномерные конусы отказа (RDO, Wollschläger et al.) через rdo.py форка.

rdo.py запускается отдельным процессом (он написан как скрипт). Шаблон чата и refusal-токены
передаются ему из models.FAMILIES, гиперпараметры — все явно из конфига: значения по умолчанию
rdo.py на результат не влияют.
"""
import os
import subprocess
import sys

from ..artifacts import FORK

# Все гиперпараметры rdo.py, которые влияют на обучение; конфиг обязан задать каждый.
HPARAMS = ("dtype", "epochs", "lr", "batch_size", "effective_batch_size", "patience", "n_lr_reduce",
           "n_sample", "fixed_samples", "sampling_method", "ablation_lambda", "addition_lambda",
           "retain_lambda", "target_generation_batch_size", "filter_data", "filter_batch_size")


def command(model, fam, domain, dim_dir, splits_dir, targets_dir, out_dir, seeds, hparams):
    """Аргументы запуска rdo.py для одного домена и списка сидов (одномерные конусы)."""
    missing = [k for k in HPARAMS if k not in hparams]
    extra = [k for k in hparams if k not in HPARAMS]
    assert not missing and not extra, f"rdo.hparams: нет {missing}, лишние {extra}"
    cmd = [sys.executable, "rdo.py", "--model", model, "--splits", domain,
           "--dim_dir", dim_dir, "--splits_dir", splits_dir, "--targets_dir", targets_dir,
           "--out_dir", out_dir, "--seeds", *map(str, seeds),
           "--chat_template", fam["template"], "--refusal_toks", *map(str, fam["refusal_toks"])]
    for k in HPARAMS:
        v = hparams[k]
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")       # store_true
            continue
        cmd += [f"--{k}", str(v)]
    return cmd


def seed_path(out_dir, seed):
    return os.path.join(out_dir, f"seed_{seed:02d}.pt")


def pending(out_dir, seeds):
    """Сиды, для которых ещё нет результата (обученные rdo.py пропускает сам, но так мы не
    грузим модель зря)."""
    return [s for s in seeds if not os.path.exists(seed_path(out_dir, s))]


def train(cmd, save_dir):
    subprocess.run(cmd, cwd=FORK, env=dict(os.environ, SAVE_DIR=save_dir), check=True)


def load_seed(out_dir, seed):
    import torch
    return torch.load(seed_path(out_dir, seed), map_location="cpu", weights_only=True)
