"""Build the BTC intraday-density parquet consumed by Gate 1.

Reads  ``data/btcusdt_5m.parquet``   (from ``wbtc fetch --timeframe 5m BTC/USDT``)
Writes ``data/btcusdt_intraday_density.parquet``  (gitignored, one row per UTC day)
       ``results/intraday_density_manifest.json`` (spec + provenance, committed)

Construction constants live in :mod:`wbtc.density` and are frozen by
``docs/PREREG.md`` v1.1. No CLI flags by design: the Gate 1 input must be
reproducible from the source parquet alone.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wbtc.density import DEFAULT_SPEC, build_daily_densities  # noqa: E402

SRC = ROOT / "data" / "btcusdt_5m.parquet"
OUT = ROOT / "data" / "btcusdt_intraday_density.parquet"
MANIFEST = ROOT / "results" / "intraday_density_manifest.json"

# Calendar exclusions (PREREG.md v1.1). BTC/USDT has no depeg days; the
# calendar is empty and kept here so the policy is explicit.
EXCLUDE_DAYS: frozenset[date] = frozenset()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not SRC.exists():
        print(
            f"ERROR: {SRC} missing; run `wbtc fetch --timeframe 5m BTC/USDT`",
            file=sys.stderr,
        )
        return 2
    ohlcv = pd.read_parquet(SRC)
    # Drop the (partial) current UTC day so the last row is a complete day.
    ts = pd.to_datetime(ohlcv["ts"], utc=True)
    today = datetime.now(timezone.utc).date()
    ohlcv = ohlcv[ts.dt.date < today].reset_index(drop=True)

    df = build_daily_densities(ohlcv, DEFAULT_SPEC, EXCLUDE_DAYS)
    df["day"] = pd.to_datetime(df["day"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    excluded = df[df["is_excluded"]]
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SRC.name,
        "source_sha256": sha256_file(SRC),
        "source_rows": int(len(ohlcv)),
        "source_span": [str(ts.min().date()), str(ohlcv["ts"].max().date())],
        "output": OUT.name,
        "output_sha256": sha256_file(OUT),
        "n_days": int(len(df)),
        "n_excluded": int(len(excluded)),
        "excluded_days": [
            {"day": str(d.date()), "reason": r}
            for d, r in zip(excluded["day"], excluded["exclusion_reason"])
        ],
        "n_obs_median": float(df["n_obs"].median()),
        "bandwidth_median": float(df["bandwidth"].median()),
        "spec": DEFAULT_SPEC.as_dict(),
        "calendar_exclusions": sorted(str(d) for d in EXCLUDE_DAYS),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[density] wrote {OUT.name}: {len(df)} days "
        f"({manifest['source_span'][0]} → {manifest['source_span'][1]}), "
        f"{len(excluded)} excluded; manifest → {MANIFEST.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
