"""Durable memory v3: a structured user profile.

    from user_profile import recall, remember, forget, as_prompt, clean

    remember({"identity": {"name": "Raja"}, "interests": ["python"]})
    recall()      # the full profile, always schema-shaped
    as_prompt()   # "identity — name: Raja; interests — python" ("" if empty)
    forget()      # wipe the file

The lineage: v1 replayed the whole transcript (memory.jsonl) — faithful
but the bill grew forever. v2 distilled flat key: value facts
(memory.json) — cheap, but flat: a second project invents project_2, and
nothing says which keys belong together. v3 gives memory a SHAPE.
labs/profile.json follows a fixed schema —

    identity     who you are        (name, location, ...)      key → value
    work         what you do        (role, company, ...)       key → value
    preferences  how you want me    (response_style, ...)      key → value
    interests    what you're into                              list, dedup

so the extractor fills a form instead of inventing structure. Dict
sections merge with new-value-wins (facts can change without growing the
file); list sections append without repeating themselves. clean() is the
bouncer: the extractor is an LLM, so nothing enters the file unchecked.

Run it directly to see the profile:

    python labs/user_profile.py
"""

import json
from pathlib import Path

PROFILE_FILE = Path(__file__).with_name("profile.json")

SECTIONS = {"identity": dict, "work": dict, "preferences": dict, "interests": list}


def _empty() -> dict:
    return {name: kind() for name, kind in SECTIONS.items()}


def clean(update) -> dict:
    """Keep only what fits the schema — known sections, scalar values,
    non-empty strings. Only sections with something real survive."""
    out = {}
    if not isinstance(update, dict):
        return out
    for section, kind in SECTIONS.items():
        val = update.get(section)
        if kind is dict and isinstance(val, dict):
            kept = {str(k): str(v).strip() for k, v in val.items()
                    if isinstance(v, (str, int, float)) and str(v).strip()}
        elif kind is list and isinstance(val, list):
            kept = [str(x).strip() for x in val
                    if isinstance(x, (str, int, float)) and str(x).strip()]
        else:
            continue
        if kept:
            out[section] = kept
    return out


def recall() -> dict:
    """The stored profile, always full-shaped — a damaged or missing file
    just means an empty profile, never a crash."""
    if not PROFILE_FILE.exists():
        return _empty()
    try:
        raw = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty()
    return {**_empty(), **clean(raw)}


def remember(update: dict) -> dict:
    """Merge a partial profile over the stored one and save. Dict sections:
    same key, new value wins. Lists: append, case-insensitively deduped —
    interests accumulate, they don't repeat."""
    profile = recall()
    for section, val in clean(update).items():
        if isinstance(val, dict):
            profile[section].update(val)
        else:
            have = {x.lower() for x in profile[section]}
            profile[section] += [x for x in val if x.lower() not in have]
    PROFILE_FILE.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return profile


def forget() -> None:
    """Delete the file — the bot meets you as a stranger next time."""
    PROFILE_FILE.unlink(missing_ok=True)


def as_prompt(profile: dict) -> str:
    """The profile as one compact line for the system prompt ('' if empty)."""
    parts = []
    for section, val in profile.items():
        if not val:
            continue
        body = ", ".join(val) if isinstance(val, list) else \
            " · ".join(f"{k}: {v}" for k, v in val.items())
        parts.append(f"{section} — {body}")
    return "; ".join(parts)


if __name__ == "__main__":
    profile = recall()
    if not any(profile.values()):
        print(f"no profile yet — {PROFILE_FILE} is empty")
    for section, val in profile.items():
        if not val:
            continue
        print(f"{section}:")
        if isinstance(val, list):
            for x in val:
                print(f"    - {x}")
        else:
            for k, v in val.items():
                print(f"    {k:>16} : {v}")
