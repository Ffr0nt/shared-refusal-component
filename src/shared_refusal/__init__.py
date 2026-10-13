"""Переиспользуемая часть пайплайна. Этапы обмениваются DirectionSet (artifacts.py).

  config     — чтение конфига и хэш;
  artifacts  — папка запуска, manifest, json, DirectionSet;
  geometry   — математика над векторами: нормировка, спектр, нулевой уровень, KL;
  data       — этап 0: источник harmful (HF-датасет по колонкам из конфига), harmless, сплиты;
  models     — всё, что зависит от семейства модели: шаблон, refusal-токены, аблация, логиты;
  directions — этап 1: DIM и RDO (единственное место, где вызывается форк) → DirectionSet;
  component  — этап 2: отбор из набора и общая компонента;
  ablation   — этап 3: ASR/KL при аблации любого набора направлений;
  analysis   — этап 4: свойства компоненты (доля энергии, контроль старта, …);
  report     — markdown-отчёты.

Слои: geometry/data/config ← artifacts/report ← component/analysis (CPU) ; models ← ablation,
directions (GPU). Шаги scripts/00–04 — тонкие сценарии поверх этих модулей.
"""
