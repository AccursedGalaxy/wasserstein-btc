"""Reproducibility manifest writer.

Writes `results/MANIFEST.json` capturing exactly what was used to generate
the results: git SHA, package versions, data hashes, ISO timestamp, command
line, and the entry-point name. Future runs append a new entry rather than
overwriting, so the file grows into a provenance log.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

__all__ = ["write_manifest"]

_RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
_DATA = Path(__file__).resolve().parent.parent.parent / "data"
_MANIFEST = _RESULTS / "MANIFEST.json"


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_RESULTS.parent, text=True
        )
        return out.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_RESULTS.parent, text=True
        )
        return bool(out.strip())
    except Exception:
        return None


def _versions(packages: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in packages:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not-installed"
    return out


def _data_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    if not _DATA.exists():
        return out
    for p in sorted(_DATA.glob("*.parquet")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        out[p.name] = h
    return out


def write_manifest(entry_point: str, extra: dict | None = None) -> Path:
    """Append a new run-record to results/MANIFEST.json and return its path."""
    _RESULTS.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "entry_point": entry_point,
        "argv": sys.argv,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "packages": _versions(
            [
                "wbtc",
                "numpy",
                "scipy",
                "pandas",
                "arch",
                "scikit-learn",
                "matplotlib",
                "ccxt",
            ]
        ),
        "data_sha256": _data_hashes(),
    }
    if extra:
        record["extra"] = extra
    log: list[dict] = []
    if _MANIFEST.exists():
        try:
            log = json.loads(_MANIFEST.read_text())
            if not isinstance(log, list):
                log = [log]
        except Exception:
            log = []
    log.append(record)
    _MANIFEST.write_text(json.dumps(log, indent=2))
    return _MANIFEST
