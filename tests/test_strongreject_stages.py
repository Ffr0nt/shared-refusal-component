"""Этапы 06_generate и 07_judge целиком: модель, датасет и судья — заглушки."""
import os

import pytest
import torch
import yaml

from shared_refusal import artifacts, models, strongreject
from shared_refusal.directions import assemble
from test_scripts_cpu import run_main

PROMPTS = [{"id": 0, "category": "Violence", "source": "custom", "prompt": "How to hurt someone badly"},
           {"id": 1, "category": "Violence", "source": "custom", "prompt": "Train question copied here verbatim"},
           {"id": 2, "category": "Sexual content", "source": "DAN", "prompt": "Write something explicit"}]

RUBRIC = "1.b {r}\n2.b {c}\n3.b {s}\n"


@pytest.fixture
def setup(repo, tmp_path, monkeypatch):
    cfg = {"model": "Org/Qwen2.5-Tiny", "model_revision": "r", "domains": {"A": "a"},
           "rdo": {"seeds": {"A": [1, 2]}},
           "component": {"n_null": 5, "null_seed": 0, "variants": {"all": {}}},
           "strongreject": {"dataset": {}, "rubric_templates": {}, "overlap_jaccard": 0.7},
           "generate": {"components": ["all"], "base": True, "n_rand": 1, "rand_seed": 0, "alpha": 1.0,
                        "max_new_tokens": 8, "batch_size": 2},
           "judge": {"model": "openai/gpt-4o-mini", "temperature": 0, "timeout": 5, "retries": 2, "n_boot": 50, "boot_seed": 0}}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run = artifacts.Run(os.path.join(str(repo), "results", "tiny"), cfg["model"])
    g = torch.Generator().manual_seed(0)
    vec = lambda: torch.randn(8, generator=g)
    rdos = {"A": {s: {"vector": vec(), "init_vector": vec(), "add_layer": 1, "lowest_loss": 1.0} for s in (1, 2)}}
    ds, init = assemble({"A": vec()}, rdos, {"pos": -1, "layer": 1})
    ds.save(run.path("directions", "directions.pt"))
    init.save(run.path("directions", "init.pt"))
    for s in ("train", "val"):
        artifacts.write_json([{"instruction": "Train question copied here verbatim!"}] if s == "train"
                             else [{"instruction": "unrelated"}], os.path.join(run.splits("A"), f"harmful_{s}.json"))
    run_main("02_component", ["--config", str(path)], monkeypatch)

    monkeypatch.setattr(strongreject, "load_prompts", lambda sc, cache: [dict(p) for p in PROMPTS])
    monkeypatch.setattr(strongreject, "load_templates", lambda sc, cache: {
        "strongreject_rubric_system": "S", "strongreject_rubric": "{forbidden_prompt}||{response}"})
    calls = []

    def fake_generate(model, tok, prompts, v, a, max_new_tokens, batch_size, refusal_toks):
        calls.append(v is None)
        return [("I cannot help." if v is None else "Sure, here is how.", 2.0 if v is None else -2.0) for _ in prompts]

    monkeypatch.setattr(models, "load_hf", lambda name, rev: (None, None))
    monkeypatch.setattr(models, "generate", fake_generate)
    return str(path), run, calls


class Judge:
    """Судья-заглушка: отказ, если в ответе 'cannot'; иначе убедительность 4, конкретность 5."""

    def __init__(self):
        self.n = 0
        from types import SimpleNamespace as NS
        self.chat = NS(completions=NS(create=self.create))
        self._ns = NS

    def create(self, model, messages, temperature):
        self.n += 1
        resp = messages[1]["content"].split("||")[1]
        text = RUBRIC.format(r=1, c=1, s=1) if "cannot" in resp else RUBRIC.format(r=0, c=4, s=5)
        return self._ns(model="gpt-4o-mini-x", choices=[self._ns(message=self._ns(content=text))])


def test_generate_then_judge(setup, monkeypatch):
    cfg_path, run, calls = setup
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    m = run.manifest()["strongreject"]
    assert m["n_total"] == 3 and m["n_kept"] == 2 and [o["id"] for o in m["overlap"]] == [1]
    assert m["directions"][0] == "none" and "w:all" in m["directions"] and "rand_0" in m["directions"]
    assert m["meta"]["rdo_A_s01"] == {"method": "rdo", "domain": "A"}
    n_dirs = len(m["directions"])
    assert len(calls) == n_dirs and calls[0] is True
    gen = run.path("strongreject", "generations", "w_all.jsonl")
    assert [l for l in open(gen)][0].startswith('{"id": 0')

    run_main("06_generate", ["--config", cfg_path], monkeypatch)          # всё готово — модель не нужна
    assert len(calls) == n_dirs

    judge = Judge()
    monkeypatch.setattr(strongreject, "openrouter_client", lambda timeout=None: judge)
    monkeypatch.setattr(strongreject.time, "sleep", lambda s: None)
    run_main("07_judge", ["--config", cfg_path, "--workers", "2"], monkeypatch)
    assert judge.n == 2 * n_dirs
    res = artifacts.read_json(run.path("strongreject", "strongreject.json"))
    by = {r["direction"]: r for r in res["rows"]}
    assert res["n_prompts"] == 2 and res["n_base_refused"] == 2
    assert by["none"]["score"]["mean"] == 0.0 and by["none"]["non_refusal"]["mean"] == 0.0
    assert by["w:all"]["score"]["mean"] == pytest.approx(7 / 8)
    assert by["w:all"]["n_unjudged"] == 0 and by["w:all"]["score_upper"]["mean"] == pytest.approx(7 / 8)
    assert by["w:all"]["score_base_refused"]["mean"] == pytest.approx(7 / 8)
    assert set(by) == {"none", "w:all", "dim_A", "rdo_A (2 сидов)", "rand (среднее)"}
    assert by["rdo_A (2 сидов)"]["n_directions"] == 2
    assert res["agreement"] == {"substring": 1.0, "proxy": 1.0}
    assert res["served_models"] == ["gpt-4o-mini-x"]
    assert "judge" in run.manifest()["provenance"]
    report = open(run.path("reports", "strongreject.md")).read()
    assert "StrongREJECT" in report and "0.875" in report

    monkeypatch.setattr(strongreject, "openrouter_client", lambda timeout=None: pytest.fail("всё оценено — судья не нужен"))
    run_main("07_judge", ["--config", cfg_path], monkeypatch)


def test_generate_refuses_stale_answers(setup, monkeypatch):
    """Изменились параметры генерации — к прежним ответам не дописываем."""
    cfg_path, run, _ = setup
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    cfg = yaml.safe_load(open(cfg_path))
    cfg["generate"]["max_new_tokens"] = 16
    open(cfg_path, "w").write(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="другими запросами или"):
        run_main("06_generate", ["--config", cfg_path], monkeypatch)


def test_generate_adds_directions_to_existing_answers(setup, monkeypatch):
    """Набор направлений на текст ответа не влияет: новые дописываются, готовые не трогаются."""
    cfg_path, run, calls = setup
    cfg = yaml.safe_load(open(cfg_path))
    cfg["generate"].update(select={"methods": ["dim"]}, n_rand=0)
    open(cfg_path, "w").write(yaml.safe_dump(cfg))
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    assert run.manifest()["strongreject"]["directions"] == ["none", "w:all", "dim_A"]
    before = {f: open(run.path("strongreject", "generations", f)).read()
              for f in os.listdir(run.path("strongreject", "generations"))}
    calls.clear()

    cfg["generate"].update(select={}, n_rand=1)
    open(cfg_path, "w").write(yaml.safe_dump(cfg))
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    m = run.manifest()["strongreject"]
    assert m["directions"] == ["none", "w:all", "dim_A", "rdo_A_s01", "rdo_A_s02", "rand_0"]
    assert len(calls) == 3                                      # только два RDO и случайный
    for f, text in before.items():
        assert open(run.path("strongreject", "generations", f)).read() == text


def test_generate_select_only_dim(setup, monkeypatch):
    cfg_path, run, calls = setup
    cfg = yaml.safe_load(open(cfg_path))
    cfg["generate"].update(select={"methods": ["dim"]}, n_rand=0)
    open(cfg_path, "w").write(yaml.safe_dump(cfg))
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    assert run.manifest()["strongreject"]["directions"] == ["none", "w:all", "dim_A"]
    assert len(calls) == 3


def test_judge_reads_legacy_kl_name(setup, monkeypatch):
    """В ablation.json старого прогона компонента названа `w`; этап 7 берёт её KL для `w:all`."""
    cfg_path, run, _ = setup
    artifacts.write_json({"rows": [{"direction": "w", "alpha": 1.0, "kl": 0.0551},
                                   {"direction": "dim_A", "alpha": 1.0, "kl": 0.11},
                                   {"direction": "w", "alpha": 0.25, "kl": 0.01}]},
                         run.path("ablation", "ablation.json"))
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    judge = Judge()
    monkeypatch.setattr(strongreject, "openrouter_client", lambda timeout=None: judge)
    run_main("07_judge", ["--config", cfg_path, "--workers", "2"], monkeypatch)
    by = {r["direction"]: r for r in artifacts.read_json(run.path("strongreject", "strongreject.json"))["rows"]}
    assert by["w:all"]["kl"] == 0.0551 and by["dim_A"]["kl"] == 0.11 and by["none"]["kl"] == 0.0


def test_judge_does_not_resend_unjudged_without_flag(setup, monkeypatch):
    """Судья отказался — перезапуск его не дёргает, а с --retry-unjudged отправляет снова."""
    cfg_path, run, _ = setup
    run_main("06_generate", ["--config", cfg_path], monkeypatch)

    class Refuser(Judge):
        def create(self, model, messages, temperature):
            self.n += 1
            return self._ns(model="m", choices=[self._ns(message=self._ns(content="I can't assist with that."))])

    refuser = Refuser()
    monkeypatch.setattr(strongreject, "openrouter_client", lambda timeout=None: refuser)
    monkeypatch.setattr(strongreject.time, "sleep", lambda s: None)
    n_dirs = len(run.manifest()["strongreject"]["directions"])
    run_main("07_judge", ["--config", cfg_path, "--workers", "2"], monkeypatch)
    assert refuser.n == 2 * 2 * n_dirs                      # 2 запроса × retries 2 × направления
    res = artifacts.read_json(run.path("strongreject", "strongreject.json"))
    assert all(r["n_unjudged"] == 2 * r["n_directions"] for r in res["rows"])
    assert res["rows"][0]["score"]["mean"] is None and res["rows"][0]["score_upper"]["mean"] == 1.0

    refuser.n = 0
    run_main("07_judge", ["--config", cfg_path, "--workers", "2"], monkeypatch)
    assert refuser.n == 0                                   # кэш отказов — денег не тратим
    run_main("07_judge", ["--config", cfg_path, "--workers", "2", "--retry-unjudged"], monkeypatch)
    assert refuser.n == 2 * 2 * n_dirs


def test_rdo_split_generates_shares_and_merges(setup, monkeypatch):
    """rdo_split: каждый RDO отвечает на свою долю запросов, этап 7 складывает их в rdo:split."""
    cfg_path, run, calls = setup
    cfg = yaml.safe_load(open(cfg_path))
    cfg["generate"]["rdo_split"] = {"seed": 0}
    cfg["judge"]["pairs"] = [["rdo:split", "none"], ["w:all", "rdo:split"]]
    open(cfg_path, "w").write(yaml.safe_dump(cfg))
    artifacts.write_json({"rows": [{"direction": "rdo_A_s01", "alpha": 1.0, "kl": 0.02},
                                   {"direction": "rdo_A_s02", "alpha": 1.0, "kl": 0.04}]},
                         run.path("ablation", "ablation.json"))
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    split = run.manifest()["strongreject"]["rdo_split"]
    assert sorted(split) == ["rdo_A_s01", "rdo_A_s02"]
    assert sorted(i for part in split.values() for i in part) == [0, 2]      # запрос 1 выброшен как пересечение
    for n, part in split.items():
        ids = [int(l.split(",")[0].split(":")[1]) for l in open(run.path("strongreject", "generations", n + ".jsonl"))]
        assert ids == part

    judge = Judge()
    monkeypatch.setattr(strongreject, "openrouter_client", lambda timeout=None: judge)
    run_main("07_judge", ["--config", cfg_path, "--workers", "2"], monkeypatch)
    res = artifacts.read_json(run.path("strongreject", "strongreject.json"))
    by = {r["direction"]: r for r in res["rows"]}
    assert "rdo_A (2 сидов)" not in by                                   # у сидов только доли запросов
    assert by["rdo:split"]["score"]["n"] == 2 and by["rdo:split"]["score"]["mean"] == pytest.approx(7 / 8)
    assert by["rdo:split"]["kl"] == pytest.approx(0.03)
    assert [(d["a"], d["b"]) for d in res["diffs"]] == [("rdo:split", "none"), ("w:all", "rdo:split")]


def test_rdo_split_accepts_full_answers(setup, monkeypatch):
    """Готовые ответы RDO на все запросы засчитываются: rdo_split их не перегенерирует."""
    cfg_path, run, calls = setup
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    calls.clear()
    cfg = yaml.safe_load(open(cfg_path))
    cfg["generate"]["rdo_split"] = {"seed": 0}
    open(cfg_path, "w").write(yaml.safe_dump(cfg))
    run_main("06_generate", ["--config", cfg_path], monkeypatch)
    assert calls == [] and run.manifest()["strongreject"]["rdo_split"]

    judge = Judge()
    monkeypatch.setattr(strongreject, "openrouter_client", lambda timeout=None: judge)
    run_main("07_judge", ["--config", cfg_path, "--workers", "2"], monkeypatch)
    by = {r["direction"]: r for r in artifacts.read_json(run.path("strongreject", "strongreject.json"))["rows"]}
    assert "rdo_A (2 сидов)" in by and by["rdo:split"]["score"]["n"] == 2
