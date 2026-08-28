"""
Hand the Python results to the Next.js dashboard.

One JSON file, written to two places: `outputs/` for the repo and
`dashboard/public/data/` for the app. The dashboard renders nothing it did
not get from here, so a screenshot of the dashboard and a screenshot of the
terminal always agree. That property is worth more than any chart on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _clean(o):
    """numpy and pandas types are not JSON, and NaN is not valid JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def write(payload: dict, out_dir: Path) -> list[Path]:
    data = _clean(payload)
    root = out_dir.parent
    targets = [out_dir / "lab.json",
               root / "dashboard" / "public" / "data" / "lab.json"]
    written = []
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(json.dumps(data, indent=2), encoding="utf-8")
        written.append(t)
    return written
