"""Tests for the public Python API and CLI smoke path."""

import json
import subprocess
import sys

import numpy as np

import wbtc
from wbtc import (
    ForecastResult,
    available_symbols,
    data_info,
    default_forecaster,
    forecast,
    load_returns,
)
from wbtc.cli import build_parser


# -------- public API --------


def test_version_is_set():
    assert isinstance(wbtc.__version__, str)
    assert wbtc.__version__.count(".") >= 1


def test_available_symbols_returns_list():
    syms = available_symbols()
    assert isinstance(syms, list)
    # tests should not depend on data being fetched, but if BTC was fetched
    # it should appear in canonical form.
    for s in syms:
        assert "/" in s


def test_default_forecaster_branches_on_horizon():
    f1 = default_forecaster(1)
    f5 = default_forecaster(5)
    assert type(f1).__name__ == "WassersteinGeodesicGated"
    assert type(f5).__name__ == "WassersteinGeodesicTheilSen"


def test_forecast_endtoend_smoke_when_data_exists(tmp_path):
    syms = available_symbols()
    if not syms:
        # nothing to test against -- skip cleanly.
        return
    sym = syms[0]
    fc = forecast(sym, horizon=5)
    assert isinstance(fc, ForecastResult)
    assert fc.quantile_levels.shape == fc.quantile_values.shape
    assert np.all(np.diff(fc.quantile_values) >= -1e-9), (
        "forecast must be monotone in u"
    )
    # the dict form must be JSON-serialisable
    d = fc.to_dict()
    s = json.dumps(d)
    assert "median" in s and "quantile_values" in s
    # central quantile interp should equal median
    assert abs(fc.median - fc.quantile(0.5)) < 1e-9


def test_data_info_has_provenance_hash_when_data_exists():
    syms = available_symbols()
    if not syms:
        return
    info = data_info(syms[0])
    assert info.n_rows > 0
    assert len(info.sha256_8) == 8
    assert "/" in info.symbol


# -------- CLI --------


def test_cli_parser_has_all_subcommands():
    p = build_parser()
    sub_actions = [a for a in p._actions if hasattr(a, "choices") and a.choices]
    assert sub_actions, "expected a subcommands action"
    choices = list(sub_actions[0].choices.keys())
    for cmd in [
        "info",
        "fetch",
        "forecast",
        "backtest",
        "backtest-long",
        "sweep",
        "test",
    ]:
        assert cmd in choices, f"missing subcommand: {cmd}"


def test_cli_version_runs():
    out = subprocess.run(
        [sys.executable, "-m", "wbtc.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert out.returncode == 0, out.stderr
    assert "wbtc" in out.stdout


def test_cli_help_runs():
    out = subprocess.run(
        [sys.executable, "-m", "wbtc.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert out.returncode == 0
    assert "forecast" in out.stdout
    assert "backtest" in out.stdout


# -------- manifest --------


def test_manifest_write_appends_and_is_valid_json(tmp_path):
    from wbtc import manifest as m

    # redirect the manifest path to a tmp dir
    p_target = tmp_path / "MANIFEST.json"
    original = m._MANIFEST
    m._MANIFEST = p_target  # type: ignore[attr-defined]
    try:
        m.write_manifest("unit-test", extra={"k": "v"})
        m.write_manifest("unit-test-2")
        log = json.loads(p_target.read_text())
        assert isinstance(log, list)
        assert len(log) == 2
        assert log[0]["entry_point"] == "unit-test"
        assert log[1]["entry_point"] == "unit-test-2"
        assert "python" in log[0]
        assert "packages" in log[0]
    finally:
        m._MANIFEST = original  # type: ignore[attr-defined]
