# Security policy

`wasserstein-btc` is research code. It does not handle credentials,
private keys, or user data, and it does not execute network requests
beyond reading public OHLCV from Binance via [`ccxt`](https://github.com/ccxt/ccxt).

## Supported versions

The latest tagged release on the `master` branch is the only supported
version. See [`CHANGELOG.md`](CHANGELOG.md) for the release history.

## Reporting a vulnerability

If you discover a vulnerability — for example a dependency exposing a
known CVE that meaningfully affects this codebase, a path-traversal or
arbitrary-file-read issue in the data loaders, or a vulnerability in
the published wheel — please **do not** open a public issue. Instead:

1. Use GitHub's private vulnerability reporting:
   [Report a vulnerability](https://github.com/AccursedGalaxy/wasserstein-btc/security/advisories/new).
2. Or email `robinbohrer7@gmail.com` with subject `wasserstein-btc security`.

Please include reproduction steps, the affected version, and (if you
have one) a suggested patch. You'll receive an acknowledgement within
72 hours. Coordinated disclosure timelines are negotiated case by case.

## Out of scope

- Forecasting quality. The model's claims are explicitly falsifiable —
  see [`docs/THEORY.md §4`](docs/THEORY.md) and the corresponding
  verdict tables in [`docs/RESULTS_LONG.md`](docs/RESULTS_LONG.md). To
  challenge a numerical claim, please open a regular issue or a
  replication report (`.github/ISSUE_TEMPLATE/replication_report.yml`),
  not a security advisory.
- Trading or risk-management advice. This is research code and is not
  a production risk system. See the disclaimer in [`README.md`](README.md).
