import os
import sys
from types import SimpleNamespace

import pytest
import torch

from shared_refusal import models
from shared_refusal.directions import assemble, dim, rdo

HP = {"dtype": "bfloat16", "epochs": 1, "lr": 0.01, "batch_size": 1, "effective_batch_size": 16, "patience": 5,
      "n_lr_reduce": 2, "n_sample": 8, "fixed_samples": 8, "sampling_method": "hypersphere",
      "ablation_lambda": 1, "addition_lambda": 0.2, "retain_lambda": 1,
      "target_generation_batch_size": 512, "filter_data": True, "filter_batch_size": 32}


def seed_result(hidden, s):
    g = torch.Generator().manual_seed(s)
    return {"vector": 3 * torch.randn(hidden, generator=g), "init_vector": torch.randn(hidden, generator=g),
            "seed": s, "add_layer": 27, "lowest_loss": 1.0 + s / 100}


def test_assemble_order_names_meta_and_init():
    hidden = 8
    dims = {"A": 5 * torch.randn(hidden), "B": torch.randn(hidden)}
    rdos = {"A": {1: seed_result(hidden, 1), 2: seed_result(hidden, 2)}, "B": {3: seed_result(hidden, 3)}}
    ds, init = assemble(dims, rdos, {"pos": -1, "layer": 27})

    assert ds.names == ["dim_A", "rdo_A_s01", "rdo_A_s02", "dim_B", "rdo_B_s03"]
    assert ds.meta[0] == {"method": "dim", "domain": "A", "pos": -1, "layer": 27}
    assert ds.meta[2] == {"method": "rdo", "domain": "A", "seed": 2, "layer": 27, "lowest_loss": 1.02}
    assert torch.allclose(ds.V.norm(dim=1), torch.ones(5, dtype=torch.float64))
    assert torch.allclose(ds["rdo_A_s01"], rdos["A"][1]["vector"].double() / rdos["A"][1]["vector"].double().norm())

    assert init.names == ["rdo_A_s01", "rdo_A_s02", "rdo_B_s03"]
    assert all(m["method"] == "init" for m in init.meta)
    v0 = rdos["B"][3]["init_vector"].double()
    assert torch.allclose(init["rdo_B_s03"], v0 / v0.norm())


def test_assemble_domain_without_rdo():
    ds, init = assemble({"A": torch.randn(4)}, {}, {"pos": -1, "layer": 3})
    assert ds.names == ["dim_A"] and len(init) == 0 and init.hidden == 4


def test_rdo_command_passes_family_and_all_hparams():
    fam = models.FAMILIES["qwen2.5"]
    cmd = rdo.command("Qwen/Qwen2.5-3B-Instruct", fam, "O1", "/r/dim/O1", "/r/splits/O1", "/r/rdo/O1/targets",
                      "/r/rdo/O1", [1, 2], HP)
    assert cmd[:2] == [sys.executable, "rdo.py"]
    arg = lambda k: cmd[cmd.index(k) + 1]
    assert arg("--model") == "Qwen/Qwen2.5-3B-Instruct" and arg("--splits") == "O1"
    assert arg("--dim_dir") == "/r/dim/O1" and arg("--out_dir") == "/r/rdo/O1"
    assert cmd[cmd.index("--seeds") + 1:cmd.index("--seeds") + 3] == ["1", "2"]
    assert arg("--chat_template") == fam["template"]
    assert cmd[cmd.index("--refusal_toks") + 1:cmd.index("--refusal_toks") + 3] == ["40", "2121"]
    assert arg("--lr") == "0.01" and arg("--addition_lambda") == "0.2" and arg("--sampling_method") == "hypersphere"
    assert "--filter_data" in cmd and cmd[cmd.index("--filter_data") + 1].startswith("--")   # флаг без значения


def test_rdo_command_false_flag_is_omitted():
    cmd = rdo.command("Qwen/Qwen2.5-3B-Instruct", models.FAMILIES["qwen2.5"], "O1", "d", "s", "t", "o", [1],
                      {**HP, "filter_data": False})
    assert "--filter_data" not in cmd


@pytest.mark.parametrize("change", [{"lr": None}, {"extra_param": 1}])
def test_rdo_command_requires_exactly_known_hparams(change):
    hp = {k: v for k, v in {**HP, **change}.items() if v is not None}
    with pytest.raises(AssertionError, match="rdo.hparams"):
        rdo.command("m", models.FAMILIES["qwen2.5"], "O1", "d", "s", "t", "o", [1], hp)


def test_rdo_pending_and_load_seed(tmp_path):
    out = str(tmp_path)
    torch.save(seed_result(4, 2), rdo.seed_path(out, 2))
    assert rdo.seed_path(out, 2).endswith("seed_02.pt")
    assert rdo.pending(out, [1, 2, 3]) == [1, 3]
    assert rdo.load_seed(out, 2)["seed"] == 2


def test_rdo_train_runs_in_fork_with_save_dir(monkeypatch):
    calls = []
    monkeypatch.setattr(rdo.subprocess, "run", lambda cmd, cwd, env, check: calls.append((cmd, cwd, env, check)))
    rdo.train(["x"], save_dir="/r")
    (cmd, cwd, env, check), = calls
    assert cmd == ["x"] and cwd == rdo.FORK and env["SAVE_DIR"] == "/r" and check


def test_dim_union_and_point():
    md = {"A": torch.ones(2, 3, 4), "B": 3 * torch.ones(2, 3, 4)}
    u = dim.union(md)
    assert torch.equal(u, 2 * torch.ones(2, 3, 4))
    x = torch.arange(24.0).reshape(2, 3, 4)
    assert torch.equal(dim.at_point(x, -1, 2), x[1, 2])


class FakeTok:
    """Токенизатор-заглушка: id — коды символов."""
    def __call__(self, text, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) for c in text]]))


def fake_model_base(template, refusal_toks):
    tok = FakeTok()
    fn = lambda instructions: tok(template.format(instruction=instructions[0]))
    return SimpleNamespace(refusal_toks=refusal_toks, tokenizer=tok, tokenize_instructions_fn=fn)


def test_check_family_accepts_matching_fork():
    fam = models.FAMILIES["qwen2.5"]
    dim.check_family(fake_model_base(fam["template"], fam["refusal_toks"]), fam)


def test_check_family_catches_token_and_template_mismatch():
    fam = models.FAMILIES["llama-3"]
    with pytest.raises(AssertionError, match="refusal"):
        dim.check_family(fake_model_base(fam["template"], [40, 1]), fam)
    with pytest.raises(AssertionError, match="шаблон"):          # лишняя кавычка, как в llama3_model.py форка
        dim.check_family(fake_model_base('"' + fam["template"], fam["refusal_toks"]), fam)
