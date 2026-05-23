"""Tiny local server for the wbtc viewer.

Run from the repo root (or anywhere):

    uv run python viewer/serve.py            # http://localhost:8765
    uv run python viewer/serve.py --port 9000

It serves viewer/public/ statically and does not touch any other file.
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import webbrowser
from functools import partial
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent / "public"
    if not (root / "index.html").exists():
        print(f"missing {root}/index.html", file=sys.stderr)
        return 2
    if not (root / "data.json").exists():
        print(
            f"warning: {root}/data.json is missing — run\n"
            "    uv run python viewer/build_data.py\n"
            "first, otherwise the viewer will show an error.",
            file=sys.stderr,
        )

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as srv:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"\n  wbtc viewer  →  {url}\n  (Ctrl-C to stop)\n")
        if not args.no_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
