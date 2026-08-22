"""A very simple terminal chatbot on the Lab 1 backend.

    python labs/chatbot.py

Same plumbing as lab_1: the _kit client (class proxy from your .env),
streamed replies, and the running cost meter. The whole bot is one loop —
read a line, append it to the history, send the WHOLE history, stream the
answer back, repeat.

Observability, the super simple version: every request appends one JSON
record to labs/traces.jsonl (see traces.py) — timestamp, latency, tokens,
ok/error. Read it back with `/log` here, `python labs/traces.py` for a
summary report, or `jq` / pandas later.

On top of that, classification: each user message is tagged by a LOCAL
Hugging Face zero-shot model (see hf_classifier.py) into one of four
categories — code generation · sales · design · non-work related. It runs
on your machine, costs zero tokens, and the tag lands in the same trace
record, so the log answers not just "how much did it cost" but "what are
people asking for".

And durable memory, the token-efficient kind: after each reply one small
extraction call distills the user's message into a structured PROFILE —
identity · work · preferences · interests — saved in labs/profile.json
(see user_profile.py; memory.json and memory.jsonl are the earlier, flat
versions of this idea). The profile rides into the system prompt — a few
dozen tokens however long you've been chatting — so the bot knows you
across restarts without replaying transcripts. The extraction has its own
row in the /cost meter, so the price of remembering stays visible. After
each reply the token bill for that turn is printed too.

Commands:  /cost  show the meter · /log  last few trace records ·
           /facts  what the bot knows about you ·
           /reset  clear the conversation (facts survive) ·
           /forget  wipe the fact sheet ·
           /quit (or q, exit, Ctrl-D, Ctrl-C)  leave
"""

import json
import re
import time

from _kit import banner, chat, client, meter, say, stream_chat
from hf_classifier import classify_local
from traces import tail, trace
from user_profile import as_prompt, clean, forget, recall, remember

SYSTEM = (
    "You are a friendly, concise assistant chatting in a terminal. "
    "Answer in a few short sentences unless the user asks for more."
)

EXTRACT = (
    "You maintain a long-term user profile for a chatbot. From the user's "
    "message, extract only DURABLE facts — things still true next week. "
    "Reply with ONLY JSON in this exact shape, every section optional: "
    '{"identity": {"name": "..", "location": ".."}, '
    '"work": {"role": "..", "company": "..", "current_project": ".."}, '
    '"preferences": {"response_style": ".."}, '
    '"interests": ["short topic", ".."]}. '
    "Short snake_case keys, short values; {} if nothing durable. Report only "
    "what is new or changed vs the current profile: "
)


def system_msg(profile: dict) -> str:
    """The system prompt with the profile folded in — the whole trick:
    one compact line instead of a replayed transcript."""
    sheet = as_prompt(profile)
    if not sheet:
        return SYSTEM
    return SYSTEM + " Known about this user from earlier chats: " + sheet + "."


def extract_profile(cli, profile: dict, user: str) -> dict:
    """One small metered call that distills the message into profile updates.
    Memory must never break the chat — any failure just returns {}."""
    try:
        raw = chat(cli, [{"role": "system", "content": EXTRACT + json.dumps(profile)},
                         {"role": "user", "content": user}],
                   label="memory", max_tokens=200)
        start, end = raw.find("{"), raw.rfind("}")  # tolerate prose/code fences
        new = json.loads(raw[start:end + 1]) if start != -1 else {}
        return clean(new)  # schema-check it — the extractor is an LLM
    except Exception:  # noqa: BLE001
        return {}


def classify(text):
    """Tag one message with the local HF model. Telemetry only — if the
    model is missing or chokes, the chat must not care."""
    try:
        return classify_local(text)
    except Exception:  # noqa: BLE001
        return None


def type_out(delta: str) -> None:
    """Print word by word with a small pause — the typing effect. Works whether
    the proxy streams token deltas or falls back to one whole-text delta."""
    for chunk in re.split(r"(\s+)", delta):
        if not chunk:
            continue
        print(chunk, end="", flush=True)
        if not chunk.isspace():
            time.sleep(0.04)


def main() -> None:
    banner("Level 2 · AI Practitioner", "A very simple chatbot (Lab 1 backend)")
    say("[dim]Type a message · /cost meter · /log traces · /facts memory · "
        "/reset new chat · /forget wipe facts · /quit to leave[/dim]\n")
    cli = client()
    say("[dim](warming up the local classifier — first ever run downloads ~440MB)[/dim]")
    if classify("hello") is None:
        say("[dim](classifier unavailable — chat still works, traces go untagged)[/dim]")
    profile = recall()
    known = sum(len(v) for v in profile.values())
    if known:
        say(f"[dim](I know {known} things about you — /facts to see them)[/dim]")
    history = [{"role": "system", "content": system_msg(profile)}]
    turn = 0

    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            say()
            break
        if not user:
            continue
        if user.lower() in ("/quit", "quit", "exit", "q"):
            break
        if user.lower() == "/cost":
            meter.show()
            continue
        if user.lower() == "/log":
            lines = tail(3)
            if not lines:
                say("[dim](no traces yet — say something first)[/dim]")
                continue
            for line in lines:
                say(line, style="dim", markup=False)  # raw JSON — brackets aren't rich tags
            continue
        if user.lower() in ("/facts", "/profile"):
            sheet = as_prompt(profile)
            if sheet:
                say(f"[dim]({sheet})[/dim]")
            else:
                say("[dim](no profile yet — tell me something about you)[/dim]")
            continue
        if user.lower() == "/reset":
            history = [{"role": "system", "content": system_msg(profile)}]
            say("[dim](conversation cleared — the profile survives; /forget wipes it)[/dim]")
            continue
        if user.lower() == "/forget":
            forget()
            profile = recall()
            history[0] = {"role": "system", "content": system_msg(profile)}
            say("[dim](profile deleted — we've never met)[/dim]")
            continue

        history.append({"role": "user", "content": user})
        say("[bold cyan]bot >[/bold cyan] ", end="")
        turn += 1
        p0, c0 = meter.prompt_tokens, meter.completion_tokens
        t0 = time.perf_counter()
        try:
            reply = stream_chat(cli, history, label="chatbot",
                                on_delta=type_out)
        except Exception as e:  # noqa: BLE001 — surface an honest error, keep chatting
            history.pop()  # don't resend the turn that failed
            trace(turn=turn, ok=False, error=type(e).__name__,
                  latency_ms=round((time.perf_counter() - t0) * 1000),
                  msgs_sent=len(history) + 1, user=user[:200])
            say(f"\n[red]call failed ({type(e).__name__}) — try again[/red]")
            continue
        latency_ms = round((time.perf_counter() - t0) * 1000)
        pt, ct = meter.prompt_tokens - p0, meter.completion_tokens - c0
        tags = classify(user)
        trace(turn=turn, ok=True, latency_ms=latency_ms,
              msgs_sent=len(history), prompt_tokens=pt, completion_tokens=ct,
              classification=tags, user=user[:200], reply=reply[:200])
        say("\n")
        if tags:
            say(f"[dim](tagged: {tags['category']} · {tags['confidence']})[/dim]")
        if pt or ct:
            say(f"[dim](tokens: {pt} prompt + {ct} completion = {pt + ct} "
                f"· session {meter.total_tokens})[/dim]")
        else:
            say("[dim](this proxy build didn't report usage for the stream)[/dim]")
        history.append({"role": "assistant", "content": reply})
        new = extract_profile(cli, profile, user)
        if new:
            profile = remember(new)
            history[0] = {"role": "system", "content": system_msg(profile)}
            say(f"[dim](profile: {as_prompt(new)})[/dim]")

    meter.show()
    say("[bold green]bye![/bold green]")


if __name__ == "__main__":
    main()
