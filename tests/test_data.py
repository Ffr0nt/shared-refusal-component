import json
import sys
import types

import pandas as pd
import pytest

from shared_refusal import data

DC = {"name": "salad", "hf_path": "Org/Fake", "hf_config": "base_set", "hf_revision": "abc123",
      "text_col": "question", "domain_col": "1-category", "stratum_col": "3-category"}


@pytest.fixture
def fake_datasets(monkeypatch):
    """Мок datasets.load_dataset: запоминает аргументы и отдаёт маленькую таблицу."""
    calls = []
    table = pd.DataFrame({
        "question": ["q1", "q2", "q3", "dup", "dup", "q4", "other"],
        "1-category": ["Domain A", "Domain A", "Domain B", "Domain A", "Domain B", "Domain B", "Domain C"],
        "3-category": ["a1", "a2", "b1", "a1", "b1", "b2", "c1"],
    })

    class Split:
        def to_pandas(self):
            return table.copy()

    def load_dataset(path, name=None, split=None, revision=None):
        calls.append({"path": path, "name": name, "split": split, "revision": revision})
        return Split()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_load_harmful_filters_domains_drops_duplicates_and_renames(fake_datasets):
    df = data.load_harmful(DC, {"A": "Domain A", "B": "Domain B"})
    assert fake_datasets == [{"path": "Org/Fake", "name": "base_set", "split": "train", "revision": "abc123"}]
    assert list(df.columns) == ["text", "domain", "stratum"]
    assert list(df["text"]) == ["q1", "q2", "q3", "q4"]          # 'dup' выброшен целиком, C отфильтрован
    assert list(df["domain"]) == ["A", "A", "B", "B"]          # код домена, не метка источника
    assert list(df["stratum"]) == ["a1", "a2", "b1", "b2"]


def test_load_harmful_without_stratum_and_custom_split(fake_datasets):
    dc = {**DC, "stratum_col": None, "hf_split": "test"}
    df = data.load_harmful(dc, {"A": "Domain A"})
    assert fake_datasets[0]["split"] == "test"
    assert set(df["stratum"]) == {"all"}


def test_load_harmless_relative_to_fork_and_absolute(tmp_path, monkeypatch):
    rows = [{"instruction": "hi"}, {"instruction": "hello"}]
    (tmp_path / "h.json").write_text(json.dumps(rows))
    monkeypatch.setattr(data, "FORK", str(tmp_path))
    assert list(data.load_harmless("h.json")["instruction"]) == ["hi", "hello"]
    assert list(data.load_harmless(str(tmp_path / "h.json"))["instruction"]) == ["hi", "hello"]


def make_pool():
    return pd.DataFrame({"text": [f"t{i}" for i in range(100)],
                         "stratum": ["x"] * 60 + ["y"] * 30 + ["z"] * 10})


@pytest.mark.parametrize("n", [10, 33, 50, 100])
def test_stratified_sample_size_and_proportions(n):
    s = data.stratified_sample(make_pool(), n, "stratum", seed=0)
    assert len(s) == n
    assert s["text"].is_unique
    counts = s["stratum"].value_counts()
    for cat, share in {"x": 0.6, "y": 0.3, "z": 0.1}.items():
        assert abs(counts.get(cat, 0) - share * n) < 1          # largest remainder: отклонение < 1


def test_stratified_sample_deterministic_per_seed():
    a = data.stratified_sample(make_pool(), 40, "stratum", seed=0)
    b = data.stratified_sample(make_pool(), 40, "stratum", seed=0)
    c = data.stratified_sample(make_pool(), 40, "stratum", seed=1)
    assert a.equals(b)
    assert not a.equals(c)


def test_cut_gives_disjoint_consecutive_parts():
    df = pd.DataFrame({"v": range(10)})
    parts = data.cut(df, {"train": 5, "val": 3, "test": 2})
    assert list(parts) == ["train", "val", "test"]
    assert [list(p["v"]) for p in parts.values()] == [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]]


# ------------------------------------------------------- kind: paired (BeaverTails)

PC = {"name": "bt", "kind": "paired", "hf_path": "Org/Paired", "hf_revision": "rev1",
      "path": "round0/330k/train.jsonl.xz", "text_col": "prompt", "response_col": "response",
      "safe_col": "is_safe", "category_col": "category", "min_target_words": 3}
DOMAINS = {"R": ["rare"], "F": ["frequent", "also_frequent"]}


def cat(**kw):
    return {c: kw.get(c, False) for c in ("rare", "frequent", "also_frequent", "other")}


ROWS = [
    # q1: метки rare и frequent -> уходит в R (он раньше в конфиге); короткий ответ отброшен
    {"prompt": "q1", "response": "too short", "is_safe": False, "category": cat(rare=True)},
    {"prompt": "q1", "response": "a long enough answer here", "is_safe": False, "category": cat(frequent=True)},
    {"prompt": "q1", "response": "another long enough answer", "is_safe": False, "category": cat(rare=True)},
    # q2: только frequent
    {"prompt": "q2", "response": "safe answer ignored", "is_safe": True, "category": cat()},
    {"prompt": "q2", "response": "three words only", "is_safe": False, "category": cat(frequent=True)},
    # q3: ни одного подходящего по длине ответа -> берётся самый длинный
    {"prompt": "q3", "response": "x y", "is_safe": False, "category": cat(also_frequent=True)},
    {"prompt": "q3", "response": "x", "is_safe": False, "category": cat(also_frequent=True)},
    # q4: метка вне доменов конфига -> выброшен
    {"prompt": "q4", "response": "some long answer text", "is_safe": False, "category": cat(other=True)},
]


@pytest.fixture
def fake_paired(monkeypatch, tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS))
    monkeypatch.setattr(data, "download", lambda dc: str(path))
    return path


def test_paired_assigns_domain_by_config_order_and_picks_one_target(fake_paired):
    df = data.load_harmful(PC, DOMAINS)
    assert list(df.columns) == ["text", "domain", "stratum", "target"]
    got = {r.text: (r.domain, r.stratum, r.target) for r in df.itertuples(index=False)}
    assert set(got) == {"q1", "q2", "q3"}                       # q4 без известных меток
    assert got["q1"][:2] == ("R", "rare")                       # приоритет домена R над F
    assert got["q1"][2] == "a long enough answer here"          # первый достаточно длинный
    assert got["q2"] == ("F", "frequent", "three words only")
    assert got["q3"] == ("F", "also_frequent", "x y")           # запасной: самый длинный
    assert df["text"].is_unique


def test_paired_checks_categories_exist(fake_paired):
    with pytest.raises(AssertionError, match="нет категорий"):
        data.load_harmful(PC, {"X": ["missing_label"]})


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="labeled или paired"):
        data.load_harmful({**PC, "kind": "whatever"}, DOMAINS)
