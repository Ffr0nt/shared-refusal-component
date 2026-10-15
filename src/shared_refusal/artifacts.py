"""Артефакты запуска: раскладка папки results/, manifest, json и набор направлений.

Раскладка results/<запуск>/ (имя запуска — имя файла конфига):
    splits/<домен>/{harmful,harmless}_{train,val,test}.json   этап 0
    dim/<домен>/direction.pt, rdo/<домен>/seed_XX.pt         этап 1, сырые (их читает rdo.py)
    directions/directions.pt, directions/init.pt             этап 1, единый набор
    component/<вариант>/                                     этап 2
    ablation/                                                этап 3
    analysis/<вариант>/                                      этап 4
    convergence/                                             этап 5
    strongreject/{prompts.json,generations/,judge/,strongreject.json}   этапы 6–7
    reports/*.md, logs/*.log, manifest.json (с provenance каждого этапа)
"""
import json
import os
from dataclasses import dataclass, field

import torch

from .geometry import unit

# src/shared_refusal/artifacts.py -> корень репозитория (пакет ставится editable, файл остаётся на месте)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORK = os.path.abspath(os.getenv("FORK", os.path.join(REPO, "..", "geometry-of-refusal")))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class Run:
    """Папка результатов одного запуска."""

    def __init__(self, root, model):
        self.root, self.model = root, model

    @classmethod
    def from_config(cls, cfg):
        return cls(os.path.join(REPO, "results", cfg["run_name"]), cfg["model"])

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def splits(self, domain):
        return self.path("splits", domain)

    def dim(self, domain):
        return self.path("dim", domain)

    def rdo(self, domain):
        return self.path("rdo", domain)

    def component(self, variant):
        return self.path("component", variant)

    def analysis(self, variant):
        return self.path("analysis", variant)

    def manifest(self):
        p = self.path("manifest.json")
        return read_json(p) if os.path.exists(p) else {"model": self.model}

    def update_manifest(self, key, value):
        """manifest.json — что получено каждым шагом (точка извлечения, сиды, хэши сплитов)."""
        m = self.manifest()
        m[key] = value
        write_json(m, self.path("manifest.json"))

    def write_report(self, name, lines):
        p = self.path("reports", name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[отчёт: {p}]")


@dataclass
class DirectionSet:
    """Именованные единичные направления одного пространства (одна модель, одна точка).

    meta[i] — метаданные i-го направления; обязательные ключи: method ('dim', 'rdo', 'component',
    'random', …) и domain (None, если направление не принадлежит домену).
    """
    names: list
    V: torch.Tensor                       # [N, hidden], float64, единичные строки
    meta: list = field(default_factory=list)

    def __post_init__(self):
        self.V = self.V.double()
        assert self.V.ndim == 2 and len(self.V) == len(self.names), "V должен быть [N, hidden]"
        assert torch.allclose(self.V.norm(dim=1), torch.ones(len(self.V), dtype=self.V.dtype), atol=1e-6), \
            "строки V должны быть единичными — используй DirectionSet.from_raw"
        if not self.meta:
            self.meta = [{} for _ in self.names]
        assert len(self.names) == len(self.V) == len(self.meta), "names, V и meta разной длины"
        assert len(set(self.names)) == len(self.names), "имена направлений должны быть уникальны"

    @classmethod
    def from_raw(cls, names, vectors, meta=None, hidden=None):
        """Из сырых векторов любой длины: каждый нормируется в float64. hidden нужен только
        для пустого набора."""
        V = torch.stack([unit(v.double()) for v in vectors]) if len(vectors) else torch.empty(0, hidden or 0)
        return cls(list(names), V, meta or [])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, name):
        return self.V[self.names.index(name)]

    @property
    def hidden(self):
        return self.V.shape[1]

    def select(self, methods=None, domains=None, exclude=()):
        """Подмножество: по методам, доменам и явному списку исключаемых имён (порядок сохраняется)."""
        unknown = set(exclude) - set(self.names)
        assert not unknown, f"исключаемых направлений нет в наборе: {sorted(unknown)}"
        keep = [i for i, (n, m) in enumerate(zip(self.names, self.meta))
                if (methods is None or m.get("method") in methods)
                and (domains is None or m.get("domain") in domains)
                and n not in exclude]
        return DirectionSet([self.names[i] for i in keep], self.V[keep].reshape(len(keep), self.hidden),
                            [self.meta[i] for i in keep])

    def where(self, method):
        return [i for i, m in enumerate(self.meta) if m.get("method") == method]

    def __add__(self, other):
        return DirectionSet(self.names + other.names, torch.cat([self.V, other.V.to(self.V)]), self.meta + other.meta)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"names": self.names, "V": self.V, "meta": self.meta}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(d["names"], d["V"], d["meta"])
