"""Episodic memory: whole conversations, filed by theme.

    from episodes import (working_append, working_messages, working_header,
                          working_clear, list_episodes, load_episode,
                          save_episode, find_episode, pointers)

Three memories now, three lifetimes:

    working    the conversation happening RIGHT NOW — labs/memory.jsonl,
               append-only, resumed on restart (a crash-proof scratchpad)
    episodic   whole past conversations — one JSON file per theme in
               labs/episodes/, transcript + a summary of its last state
    durable    the distilled user profile (user_profile.py)

The working buffer is the v1 memory file with a new job: it holds ONLY the
current episode. When the topic changes and the user agrees to switch, the
whole buffer moves into labs/episodes/<id>-<slug>.json — along with a
fresh summary of where things stood — and the buffer restarts: empty for
a new episode, or preloaded with the transcript of an old episode being
resumed. A header line in the buffer ({"episode": id, "title": ...})
remembers which episode the scratchpad belongs to, so a restart drops you
exactly where you were.

Run it directly to see the shelf:

    python labs/episodes.py
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

WORKING_FILE = Path(__file__).with_name("memory.jsonl")
EPISODES_DIR = Path(__file__).with_name("episodes")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── the working buffer (the conversation happening now) ─────────────────────

def working_append(role: str, content: str) -> None:
    """Append one message to the scratchpad."""
    with WORKING_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), "role": role, "content": content},
                           ensure_ascii=False) + "\n")


def _working_lines() -> list:
    if not WORKING_FILE.exists():
        return []
    out = []
    for line in WORKING_FILE.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a torn line can't take the chat down
            continue
    return out


def working_messages() -> list:
    """The buffered conversation, ready for the messages list."""
    return [{"role": r["role"], "content": r["content"]}
            for r in _working_lines() if "role" in r]


def working_header() -> dict:
    """Which episode this buffer belongs to ({} if untitled) — header lines
    carry an "episode" key instead of a "role"; the last one written wins."""
    header = {}
    for r in _working_lines():
        if "episode" in r:
            header = r
    return header


def working_clear(episode_id=None, title=None, messages=()) -> None:
    """Restart the scratchpad — empty for a fresh episode, or preloaded
    with a resumed episode's transcript."""
    WORKING_FILE.unlink(missing_ok=True)
    if episode_id or title:
        with WORKING_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), "episode": episode_id,
                                "title": title}, ensure_ascii=False) + "\n")
    for m in messages:
        working_append(m["role"], m["content"])


# ── the episode shelf ────────────────────────────────────────────────────────

def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (title or "episode").lower()).strip("-")[:40] or "episode"


def list_episodes() -> list:
    """Every episode on the shelf, oldest-updated first."""
    if not EPISODES_DIR.exists():
        return []
    eps = []
    for path in EPISODES_DIR.glob("*.json"):
        try:
            eps.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    eps.sort(key=lambda e: e.get("updated", ""))
    return eps


def load_episode(eid: str):
    for ep in list_episodes():
        if ep["id"] == eid:
            return ep
    return None


def find_episode(query: str):
    """An episode by exact id, or the newest whose title contains the words."""
    if not query:
        return None
    q = query.strip().lower()
    hit = None
    for ep in list_episodes():          # oldest→newest, so the last hit is newest
        if ep["id"] == q or q in ep["title"].lower():
            hit = ep
    return hit


def save_episode(title: str, summary: str, messages: list, eid=None) -> dict:
    """File a transcript on the shelf. No eid → a new episode; an existing
    eid → that episode updated in place (re-filed after being resumed)."""
    EPISODES_DIR.mkdir(exist_ok=True)
    eps = list_episodes()
    old = load_episode(eid) if eid else None
    if not old:
        nums = [int(e["id"]) for e in eps if str(e["id"]).isdigit()]
        eid = f"{(max(nums) + 1) if nums else 1:03d}"
    ep = {"id": eid, "title": title,
          "created": old["created"] if old else _now(), "updated": _now(),
          "summary": summary, "messages": list(messages)}
    path = EPISODES_DIR / f"{eid}-{_slug(title)}.json"
    for stale in EPISODES_DIR.glob(f"{eid}-*.json"):  # title may have changed
        if stale != path:
            stale.unlink()
    path.write_text(json.dumps(ep, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return ep


def pointers(n: int = 3, skip_id=None) -> str:
    """The last few episodes as one compact line for the system prompt —
    the bot's awareness that other conversations exist."""
    eps = [e for e in list_episodes() if e["id"] != skip_id][-n:]
    parts = []
    for e in eps:
        summary = e["summary"][:90] + "…" if len(e["summary"]) > 90 else e["summary"]
        parts.append(f"[{e['id']}] {e['title']}: {summary}")
    return "; ".join(parts)


if __name__ == "__main__":
    eps = list_episodes()
    if not eps:
        print(f"no episodes yet — {EPISODES_DIR}/ is empty")
    for e in eps:
        print(f"[{e['id']}] {e['title']}  ·  {len(e['messages'])} messages  ·  updated {e['updated']}")
        print(f"      {e['summary']}")
    header = working_header()
    msgs = working_messages()
    if msgs:
        where = f"[{header.get('episode')}] {header.get('title')}" if header else "(untitled)"
        print(f"\nworking buffer: {len(msgs)} messages in {where}")
