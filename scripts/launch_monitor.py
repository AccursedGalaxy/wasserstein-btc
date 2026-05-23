"""Poll HN + Reddit + GitHub for new activity on launch-day URLs.

Pure stdlib, no deps — runs under any Python ≥3.8. Idempotent.

State lives at .launch/state.json. To watch a new URL, append to the
`watch.hn` or `watch.reddit` arrays (or use the --add helpers).

Outputs a JSON summary to stdout. Caller (the cron agent) parses it and
decides whether to ping the user via Telegram.

Usage:
    python3 scripts/launch_monitor.py                          # poll all
    python3 scripts/launch_monitor.py --add-hn 12345678        # watch HN item
    python3 scripts/launch_monitor.py --add-reddit URL         # watch reddit
    python3 scripts/launch_monitor.py --status                 # show watchlist
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / ".launch" / "state.json"
REPO_FULL_NAME = "AccursedGalaxy/wasserstein-btc"

HN_API = "https://hacker-news.firebaseio.com/v0"
UA = "wbtc-launch-monitor/1.0 (https://github.com/AccursedGalaxy/wasserstein-btc)"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _http_get_json(url: str, timeout: int = 12) -> dict | list | None:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as e:
        return {"_error": str(e), "_url": url}


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "watch": {"hn": [], "reddit": []},
            "last": {"stars": None, "hn": {}, "reddit": {}},
            "history": [],
        }
    return json.loads(STATE_PATH.read_text())


def _save_state(s: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(s, indent=2, sort_keys=True))


# ─── HN ───────────────────────────────────────────────────────────────────


def _hn_item(item_id: str) -> dict | None:
    payload = _http_get_json(f"{HN_API}/item/{item_id}.json")
    if isinstance(payload, dict) and "_error" not in payload:
        return payload
    return None


def _hn_comment_body(comment_id: str) -> tuple[str, str, int] | None:
    """Return (author, plain-text body, posted-epoch) for an HN comment id."""
    item = _hn_item(comment_id)
    if not item or item.get("deleted") or item.get("dead"):
        return None
    text = item.get("text", "") or ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return (item.get("by", "?"), text, int(item.get("time", 0)))


def _walk_hn_comments(kids: list[int]) -> list[str]:
    """Return a flat list of comment IDs (top-level only — we keep
    the per-comment reply tree small for now)."""
    return [str(k) for k in (kids or [])]


def _poll_hn_item(item_id: str, last: dict) -> dict:
    item = _hn_item(item_id)
    if not item:
        return {"id": item_id, "_error": "no_item"}
    score = int(item.get("score", 0))
    title = item.get("title") or ""
    top_kids = _walk_hn_comments(item.get("kids") or [])
    seen = set(last.get("seen_comments", []))
    new_ids = [k for k in top_kids if k not in seen]
    new_comments = []
    for cid in new_ids[:20]:
        bod = _hn_comment_body(cid)
        if bod is None:
            continue
        author, text, ts = bod
        new_comments.append(
            {
                "id": cid,
                "author": author,
                "text": text[:1500],
                "posted_at": dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "url": f"https://news.ycombinator.com/item?id={cid}",
            }
        )
    result = {
        "id": item_id,
        "title": title,
        "url": f"https://news.ycombinator.com/item?id={item_id}",
        "score": score,
        "descendants": int(item.get("descendants", 0)),
        "score_prev": last.get("score"),
        "new_comments": new_comments,
    }
    return result


# ─── Reddit ───────────────────────────────────────────────────────────────


def _reddit_json_url(post_url: str) -> str:
    u = post_url.rstrip("/")
    if not u.endswith(".json"):
        u += ".json"
    return u


def _flatten_reddit_comments(listing: list) -> list[dict]:
    """Walk Reddit's comment tree, return a flat list of {id, author, body, ts}."""
    out = []

    def walk(things: list):
        for it in things:
            if not isinstance(it, dict):
                continue
            kind = it.get("kind")
            data = it.get("data") or {}
            if kind == "t1":
                out.append(
                    {
                        "id": data.get("id"),
                        "author": data.get("author"),
                        "body": (data.get("body") or "")[:1500],
                        "ts": data.get("created_utc"),
                        "permalink": f"https://www.reddit.com{data.get('permalink', '')}",
                    }
                )
                replies = data.get("replies")
                if isinstance(replies, dict):
                    children = replies.get("data", {}).get("children") or []
                    walk(children)

    if isinstance(listing, list) and len(listing) >= 2:
        children = listing[1].get("data", {}).get("children") or []
        walk(children)
    return out


def _poll_reddit(post_url: str, last: dict) -> dict:
    payload = _http_get_json(_reddit_json_url(post_url))
    if not isinstance(payload, list) or len(payload) < 2:
        return {"url": post_url, "_error": "no_payload", "_raw": payload}
    post_data = payload[0].get("data", {}).get("children", [{}])[0].get("data", {})
    score = int(post_data.get("score", 0))
    n_comments = int(post_data.get("num_comments", 0))
    title = post_data.get("title", "")
    subreddit = post_data.get("subreddit", "")
    seen = set(last.get("seen_comments", []))
    all_c = _flatten_reddit_comments(payload)
    new = [c for c in all_c if c["id"] and c["id"] not in seen]
    return {
        "url": post_url,
        "title": title,
        "subreddit": subreddit,
        "score": score,
        "score_prev": last.get("score"),
        "num_comments": n_comments,
        "new_comments": new[:20],
    }


# ─── GitHub stars ─────────────────────────────────────────────────────────


def _poll_stars() -> dict:
    try:
        out = subprocess.run(
            [
                "gh",
                "api",
                f"/repos/{REPO_FULL_NAME}",
                "--jq",
                ".stargazers_count,.forks_count,.subscribers_count",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return {"_error": out.stderr.strip()[:200]}
        lines = out.stdout.strip().splitlines()
        if len(lines) >= 3:
            return {
                "stars": int(lines[0]),
                "forks": int(lines[1]),
                "watchers": int(lines[2]),
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        return {"_error": str(e)}
    return {"_error": "unparseable"}


# ─── Main ─────────────────────────────────────────────────────────────────


def cmd_poll(state: dict) -> dict:
    summary: dict = {
        "polled_at": _now_iso(),
        "hn": [],
        "reddit": [],
        "stars": None,
        "anything_new": False,
    }

    # GitHub stars
    stars_now = _poll_stars()
    if "stars" in stars_now:
        last_stars = state["last"].get("stars")
        delta = (stars_now["stars"] - last_stars) if last_stars is not None else 0
        summary["stars"] = {**stars_now, "delta": delta, "prev": last_stars}
        if last_stars is None or delta > 0:
            summary["anything_new"] = True
        state["last"]["stars"] = stars_now["stars"]

    # HN
    for w in state["watch"]["hn"]:
        item_id = w["id"]
        prev = state["last"]["hn"].get(item_id, {})
        r = _poll_hn_item(item_id, prev)
        summary["hn"].append(r)
        if r.get("new_comments"):
            summary["anything_new"] = True
        if r.get("score") and prev.get("score") != r.get("score"):
            summary["anything_new"] = True
        seen = set(prev.get("seen_comments", []))
        for c in r.get("new_comments", []):
            seen.add(c["id"])
        state["last"]["hn"][item_id] = {
            "score": r.get("score"),
            "descendants": r.get("descendants"),
            "seen_comments": sorted(seen),
        }
        time.sleep(0.4)

    # Reddit
    for w in state["watch"]["reddit"]:
        url = w["url"]
        prev = state["last"]["reddit"].get(url, {})
        r = _poll_reddit(url, prev)
        summary["reddit"].append(r)
        if r.get("new_comments"):
            summary["anything_new"] = True
        if r.get("score") and prev.get("score") != r.get("score"):
            summary["anything_new"] = True
        seen = set(prev.get("seen_comments", []))
        for c in r.get("new_comments", []):
            seen.add(c["id"])
        state["last"]["reddit"][url] = {
            "score": r.get("score"),
            "num_comments": r.get("num_comments"),
            "seen_comments": sorted(seen),
        }
        time.sleep(1.2)  # be kind to reddit

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--add-hn", help="HN item id to watch (e.g. 12345678)")
    ap.add_argument("--add-reddit", help="Full reddit post URL to watch")
    ap.add_argument(
        "--status", action="store_true", help="Show watchlist and last-state"
    )
    args = ap.parse_args()

    state = _load_state()

    if args.add_hn:
        item_id = re.sub(r".*id=", "", args.add_hn).strip()
        if any(w["id"] == item_id for w in state["watch"]["hn"]):
            print(f"already watching HN {item_id}", file=sys.stderr)
        else:
            state["watch"]["hn"].append({"id": item_id, "added": _now_iso()})
            _save_state(state)
            print(f"added HN {item_id} to watchlist", file=sys.stderr)
        return 0

    if args.add_reddit:
        url = args.add_reddit.strip().rstrip("/")
        if any(w["url"] == url for w in state["watch"]["reddit"]):
            print(f"already watching reddit {url}", file=sys.stderr)
        else:
            state["watch"]["reddit"].append({"url": url, "added": _now_iso()})
            _save_state(state)
            print(f"added reddit {url} to watchlist", file=sys.stderr)
        return 0

    if args.status:
        print(
            json.dumps(
                {
                    "watch": state["watch"],
                    "last_stars": state["last"].get("stars"),
                    "n_hn_watched": len(state["watch"]["hn"]),
                    "n_reddit_watched": len(state["watch"]["reddit"]),
                },
                indent=2,
            )
        )
        return 0

    summary = cmd_poll(state)

    # append a tiny history breadcrumb for daily-summary rollups later
    state.setdefault("history", []).append(
        {
            "at": summary["polled_at"],
            "stars": (summary.get("stars") or {}).get("stars"),
            "hn_scores": [
                {"id": h["id"], "score": h.get("score"), "n": h.get("descendants")}
                for h in summary["hn"]
                if "id" in h
            ],
            "reddit_scores": [
                {"url": r["url"], "score": r.get("score"), "n": r.get("num_comments")}
                for r in summary["reddit"]
                if "url" in r
            ],
        }
    )
    # keep history bounded
    state["history"] = state["history"][-500:]

    _save_state(state)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
