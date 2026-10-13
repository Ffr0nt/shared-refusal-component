import json
import sys
import types

import pandas as pd
import pytest

from shared_refusal import data

DC = {"name": "salad", "hf_path": "Org/Fake", "hf_config": "base_set",
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

    def load_dataset(path, name=None, split=None):
        calls.append({"path": path, "name": name, "split": split})
        return Split()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_load_harmful_filters_domains_drops_duplicates_and_renames(fake_datasets):
    df = data.load_harmful(DC, ["Domain A", "Domain B"])
    assert fake_datasets == [{"path": "Org/Fake", "name": "base_set", "split": "train"}]
    assert list(df.columns) == ["text", "domain", "stratum"]
    assert list(df["text"]) == ["q1", "q2", "q3", "q4"]          # 'dup' выброшен целиком, C отфильтрован
    assert list(df["domain"]) == ["Domain A", "Domain A", "Domain B", "Domain B"]
    assert list(df["stratum"]) == ["a1", "a2", "b1", "b2"]


def test_load_harmful_without_stratum_and_custom_split(fake_datasets):
    dc = {**DC, "stratum_col": None, "hf_split": "test"}
    df = data.load_harmful(dc, ["Domain A"])
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
