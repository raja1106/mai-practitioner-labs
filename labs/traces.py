"""Super simple observability: one JSON record per request.

    from traces import trace, tail

    trace(turn=1, ok=True, latency_ms=812, ...)   # appends one line
    tail(3)                                        # last 3 raw records

Every record is one line of JSON appended to labs/traces.jsonl — you append
forever and never rewrite the file, and every log tool reads it (`jq`,
`pandas.read_json(..., lines=True)`, plain `cat`). The timestamp is added
here so callers never forget it.

Run it directly for a one-screen report of what's in the log:

    python labs/traces.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

TRACE_FILE = Path(__file__).with_name("traces.jsonl")


def trace(**record) -> None:
    """Append one JSON record per request — the whole observability stack."""
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **record}
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tail(n=3):
    """The last n records as raw JSON lines (newest last)."""
    if not TRACE_FILE.exists():
        return []
    return TRACE_FILE.read_text(encoding="utf-8").splitlines()[-n:]


def report() -> None:
    """Aggregate the log: counts, latency, tokens, and the category mix."""
    lines = tail(10**9)
    if not lines:
        print(f"no traces yet — {TRACE_FILE} is empty")
        return
    recs = [json.loads(line) for line in lines]
    ok = [r for r in recs if r.get("ok")]
    lat = sorted(r["latency_ms"] for r in ok if "latency_ms" in r)
    tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in ok)
    by_cat = {}
    for r in ok:
        cat = (r.get("classification") or {}).get("category", "untagged")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print(f"{len(recs)} requests · {len(recs) - len(ok)} errors · {tokens} tokens")
    if lat:
        print(f"latency p50 {lat[len(lat) // 2]}ms · worst {lat[-1]}ms")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d} × {cat}")


if __name__ == "__main__":
    report()
