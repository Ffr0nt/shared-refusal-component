from shared_refusal import report


def test_table_markdown():
    assert report.table(["a", "b"], [[1, "x"]]) == ["| a | b |", "|---|---|", "| 1 | x |"]


def test_component_report_contains_numbers():
    summary = {"lam": [0.419, 0.107, 0.068], "null": {"mean": 0.05, "p95": 0.051}, "N": 24, "hidden": 2048,
               "n_by_method": {"dim": 4, "rdo": 20}, "cos": {"dim_O1": 0.714, "rdo_O2_s08": -0.161}}
    text = "\n".join(report.component("Qwen", "all", {}, summary))
    assert "вариант `all`" in text and "24 векторов (4 DIM, 20 RDO)" in text
    assert "**0.419**" in text and "0.042" in text                    # 1/N
    assert "| rdo_O2_s08 | -0.161 | 0.026 |" in text


def test_analysis_report_handles_missing_rdo_and_init():
    lines = report.analysis("Qwen", "dim_only", 2048, {"O1": 0.5}, {}, {"O1": {"O1": 1.0}}, [])
    text = "\n".join(lines)
    assert "| O1 | **0.500** | 0.500 | — |" in text and "Контроль общего старта" not in text


def test_ablation_report_rows():
    rows = [{"direction": "w:all", "alpha": 0.25, "kl": 0.0379, "asr": {"O1": 0.922}, "asr_mean": 0.922}]
    text = "\n".join(report.ablation("Qwen", {"O1": 0.344}, rows, ["O1"]))
    assert "Без аблации: O1 0.344." in text
    assert "| w:all | 0.25 | 0.0379 | 0.922 | **0.922** |" in text


def test_component_report_with_two_vectors():
    summary = {"lam": [0.9, 0.1], "null": {"mean": 0.5, "p95": 0.6}, "N": 2, "hidden": 8,
               "n_by_method": {"dim": 2}, "cos": {"dim_A": 0.9, "dim_B": 0.9}}
    assert "| **0.900** | 0.100 | — |" in "\n".join(report.component("M", "dim_only", {"methods": ["dim"]}, summary))
