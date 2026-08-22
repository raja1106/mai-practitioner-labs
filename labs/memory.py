"""Durable memory: a small key → value fact sheet about the user.

    from memory import recall, remember, forget, as_prompt

    remember({"name": "Raja"})   # merge new facts in, save the file
    recall()                     # {"name": "Raja", ...}
    as_prompt(recall())          # "name: Raja · ..." — ready for the prompt
    forget()                     # wipe the sheet

Why key → value and not a transcript? Replaying the whole conversation
back into the prompt is faithful but the token bill grows every turn,
forever. Almost all of that text is ephemeral — what the user will still
care about NEXT WEEK fits in a handful of invariant facts (name, role,
preferences, projects). So that's what we store: labs/memory.json, one
small JSON dict, rewritten on change. The fact sheet rides into the
system prompt for a few dozen tokens no matter how long you've been
chatting.

Run it directly to see what the bot knows:

    python labs/memory.py
"""

import json
from pathlib import Path

MEMORY_FILE = Path(__file__).with_name("memory.json")


def recall() -> dict:
    """The fact sheet, or {} if nothing is known (or the file is damaged —
    memory must never crash the chat)."""
    if not MEMORY_FILE.exists():
        return {}
    try:
        facts = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return facts if isinstance(facts, dict) else {}
    except json.JSONDecodeError:
        return {}


def remember(new_facts: dict) -> dict:
    """Merge new facts over the old ones and save. Same key = the new value
    wins, so 'favorite_editor' can change without growing the file."""
    facts = recall()
    facts.update({str(k): str(v) for k, v in new_facts.items() if str(v).strip()})
    MEMORY_FILE.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return facts


def forget() -> None:
    """Delete the sheet — the bot meets you as a stranger next time."""
    MEMORY_FILE.unlink(missing_ok=True)


def as_prompt(facts: dict) -> str:
    """The sheet as one compact line for the system prompt."""
    return " · ".join(f"{k}: {v}" for k, v in facts.items())


if __name__ == "__main__":
    facts = recall()
    if not facts:
        print(f"nothing remembered yet — {MEMORY_FILE} is empty")
    for k, v in facts.items():
        print(f"{k:>20} : {v}")
