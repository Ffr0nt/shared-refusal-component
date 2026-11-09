"""directions/targets.py — готовые цели RDO, в которых нет DIM."""
import json
import os

import pytest

from shared_refusal import models
from shared_refusal.artifacts import read_json, write_json
from shared_refusal.directions import targets

FAM = models.FAMILIES["qwen2.5"]


class FakeTok:
    """Токены — слова; decode склеивает их обратно."""

    def encode(self, text, add_special_tokens=True):
        return text.split()

    def decode(self, toks, skip_special_tokens=True):
        return " ".join(toks)


@pytest.fixture
def cfg_run(repo, tmp_path):
    cfg = {"model": "Org/Qwen2.5-3B-Instruct", "model_alias": "Qwen2.5-3B-Instruct", "model_revision": "rev",
           "run_name": "bt-test", "domains": {"A": ["x"], "B": ["y"]},
           "rdo": {"targets": {"source": "dataset", "n_tokens": 3, "batch_size": 2},
                   "hparams": {"addition_lambda": 0},
                   "seeds": {"A": [1], "B": [2]}}}
    from shared_refusal.artifacts import Run
    run = Run.from_config(cfg)
    for d in cfg["domains"]:
        write_json([{"instruction": f"{d} harmful one", "target": "one two three four five"},
                    {"instruction": f"{d} harmful two", "target": "short answer"}],
                   os.path.join(run.splits(d), "harmful_train.json"))
        write_json([{"instruction": "harmless one"}, {"instruction": "harmless two"}],
                   os.path.join(run.splits(d), "harmless_train.json"))
    return cfg, run


@pytest.fixture
def fake_model(monkeypatch):
    """Модель не грузится; generate отдаёт по ответу на запрос и считает вызовы."""
    calls = []
    monkeypatch.setattr(models, "load_hf", lambda name, rev: ("MODEL", FakeTok()))
    monkeypatch.setattr(targets.models, "load_hf", lambda name, rev: ("MODEL", FakeTok()))

    def generate(model, tok, prompts, v, a, max_new_tokens, batch_size, refusal_toks):
        calls.append({"n": len(prompts), "v": v, "a": a, "max_new_tokens": max_new_tokens})
        return [(f"retain for {p[-20:]} extra words", 0.0) for p in prompts]

    monkeypatch.setattr(targets.models, "generate", generate)
    return calls


def test_truncate_keeps_first_n_tokens():
    assert targets.truncate(FakeTok(), "a b c d e", 3) == "a b c"
    assert targets.truncate(FakeTok(), "a b", 3) == "a b"


def test_write_builds_targets_without_dim(cfg_run, fake_model):
    cfg, run = cfg_run
    targets.write(cfg, run, FAM, list(cfg["domains"]))

    for d in cfg["domains"]:
        harmful = read_json(os.path.join(run.rdo(d), "targets", "harmful_targets.json"))
        harmless = read_json(os.path.join(run.rdo(d), "targets", "harmless_targets.json"))
        # порядок и число записей — как в сплитах: filter_data в rdo.py отбирает по индексам
        assert [r["ablation"] for r in harmful] == ["one two three", "short answer"]
        assert harmful[0]["prompt"] == FAM["template"].format(instruction=f"{d} harmful one")
        # цель наведения отказа пустая: через неё DIM и попадал в обучение
        assert [r["addition"] for r in harmless] == ["", ""]
        assert all(r["retain"].startswith("retain for") for r in harmless)
        assert all(len(r["retain"].split()) == 3 for r in harmless)

    # retain генерируется один раз на все домены (harmless-сплит общий), без интервенции
    assert len(fake_model) == 1
    assert fake_model[0] == {"n": 2, "v": None, "a": 0, "max_new_tokens": 3}
    assert run.manifest()["targets"]["A"] == {"harmful": 2, "harmless": 2}


def test_write_skips_done_domains(cfg_run, fake_model):
    cfg, run = cfg_run
    targets.write(cfg, run, FAM, list(cfg["domains"]))
    targets.write(cfg, run, FAM, list(cfg["domains"]))
    assert len(fake_model) == 1                       # второй раз модель не грузится вовсе
    assert targets.pending(run, list(cfg["domains"])) == []


def test_write_requires_addition_lambda_zero(cfg_run, fake_model):
    cfg, run = cfg_run
    cfg["rdo"]["hparams"]["addition_lambda"] = 0.2
    with pytest.raises(AssertionError, match="addition_lambda"):
        targets.write(cfg, run, FAM, list(cfg["domains"]))


def test_write_requires_target_field(cfg_run, fake_model):
    cfg, run = cfg_run
    write_json([{"instruction": "no target here"}], os.path.join(run.splits("A"), "harmful_train.json"))
    with pytest.raises(AssertionError, match="target"):
        targets.write(cfg, run, FAM, ["A"])


def test_write_rejects_different_harmless_splits(cfg_run, fake_model):
    cfg, run = cfg_run
    write_json([{"instruction": "other one"}, {"instruction": "other two"}],
               os.path.join(run.splits("B"), "harmless_train.json"))
    with pytest.raises(AssertionError, match="harmless_train"):
        targets.write(cfg, run, FAM, ["A", "B"])
