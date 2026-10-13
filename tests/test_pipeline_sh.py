"""scripts/pipeline.sh: проверки до запуска этапов (хэш конфига, GPU, аргументы)."""
import hashlib
import os
import subprocess

from shared_refusal.artifacts import REPO

PIPELINE = os.path.join(REPO, "scripts", "pipeline.sh")


def run(env, *args):
    e = {k: v for k, v in os.environ.items() if k not in ("CONFIG", "CONFIG_SHA256", "GPU", "STEPS")}
    return subprocess.run(["bash", PIPELINE, *args], env={**e, **env}, capture_output=True, text=True)


def test_requires_config():
    r = run({})
    assert r.returncode != 0 and "CONFIG" in r.stderr


def test_rejects_changed_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: M\n")
    r = run({"CONFIG": str(cfg), "CONFIG_SHA256": "0" * 64})
    assert r.returncode != 0 and "изменён после фиксации" in r.stderr


def test_gpu_stage_requires_gpu(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: M\n")
    sha = hashlib.sha256(cfg.read_bytes()).hexdigest()
    r = run({"CONFIG": str(cfg), "CONFIG_SHA256": sha, "STEPS": "02 03"})
    assert r.returncode != 0 and "этап 03 требует GPU" in r.stderr


def test_args_only_for_single_step(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: M\n")
    r = run({"CONFIG": str(cfg), "STEPS": "02 04"}, "--variants", "all")
    assert r.returncode != 0 and "только одному этапу" in r.stderr


def test_launch_files_pin_their_config():
    launch = os.path.join(REPO, "launch")
    for f in os.listdir(launch):
        if not f.endswith(".sh"):
            continue
        text = open(os.path.join(launch, f)).read()
        name = f[:-3]
        assert f"CONFIG=configs/{name}.yaml" in text, f
        sha = text.split("CONFIG_SHA256=")[1].split()[0]
        with open(os.path.join(REPO, "configs", f"{name}.yaml"), "rb") as c:
            assert hashlib.sha256(c.read()).hexdigest() == sha, f"{f}: хэш не совпадает с конфигом"
