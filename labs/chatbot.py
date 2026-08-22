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

Episodic memory ties it together (see episodes.py). The working memory —
labs/memory.jsonl, the v1 file with a new job — holds ONLY the current
conversation and is resumed on restart. From the second message on, a
small call checks whether your message continues the current theme; if
not, you're ASKED whether to switch episodes. Saying yes files the whole
buffer into labs/episodes/<id>-<slug>.json with a summary of its last
state, then either resumes the matching old episode (its transcript comes
back) or starts a fresh one. The system prompt carries pointers to the
last few episodes, so the bot knows what else you two have discussed.

Commands:  /cost  show the meter · /log  last few trace records ·
           /facts  what the bot knows about you ·
           /episodes  the shelf of past conversations ·
           /switch [topic]  file this chat, resume an old episode or start fresh ·
           /reset  clear conversation + working buffer (facts survive) ·
           /forget  wipe the fact sheet ·
           /quit (or q, exit, Ctrl-D, Ctrl-C)  leave
"""

import json
import re
import time

from _kit import banner, chat, client, meter, say, stream_chat
from episodes import (find_episode, list_episodes, pointers, save_episode,
                      working_append, working_clear, working_header,
                      working_messages)
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


TOPIC = (
    "You file conversations by theme. Given the user's next message, reply "
    'with ONLY JSON: {"topic": "2-4 word theme of the message", '
    '"same_episode": true, "matches_episode": "id or empty string"}. '
    "same_episode: true if the message continues the current conversation's "
    "theme, false if it clearly changes subject. matches_episode: the id of a "
    "recent episode with the same theme as the message, if any. "
)

WRAP = (
    "Summarize this conversation's LAST STATE in 2-3 sentences — what was "
    "discussed, what was decided, what threads are still open. Also give it "
    'a short title. Reply with ONLY JSON: {"title": "2-4 words", "summary": "..."}'
)


def system_msg(profile: dict, episode_title=None) -> str:
    """The system prompt with all three memories folded in: the profile
    (durable), pointers to recent episodes (episodic), and the current
    theme — one compact paragraph instead of replayed transcripts."""
    parts = [SYSTEM]
    sheet = as_prompt(profile)
    if sheet:
        parts.append("Known about this user from earlier chats: " + sheet + ".")
    eps = pointers(3)
    if eps:
        parts.append("Recent episodes (other conversations you two had — the "
                     "user can switch back to one): " + eps + ".")
    if episode_title:
        parts.append(f"The current conversation's theme: {episode_title}.")
    return " ".join(parts)


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


def detect_topic(cli, episode_title, user: str):
    """Small metered call: does this message continue the current episode's
    theme? Failure means None — and None means 'don't interrupt the chat'."""
    try:
        prompt = (TOPIC + f'Current theme: "{episode_title or "not named yet"}". '
                  f"Recent episodes: {pointers(3) or 'none'}.")
        raw = chat(cli, [{"role": "system", "content": prompt},
                         {"role": "user", "content": user}],
                   label="episodes", max_tokens=80)
        start, end = raw.find("{"), raw.rfind("}")
        det = json.loads(raw[start:end + 1]) if start != -1 else {}
        return det if isinstance(det, dict) and "same_episode" in det else None
    except Exception:  # noqa: BLE001
        return None


def file_working_episode(cli, episode_id, episode_title):
    """Move the working buffer onto the episode shelf with a fresh last-state
    summary. Returns the filed episode, or None if the buffer was empty."""
    msgs = working_messages()
    if not msgs:
        return None
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)[-6000:]
    try:
        raw = chat(cli, [{"role": "system", "content": WRAP},
                         {"role": "user", "content": transcript}],
                   label="episodes", max_tokens=200)
        start, end = raw.find("{"), raw.rfind("}")
        wrap = json.loads(raw[start:end + 1]) if start != -1 else {}
    except Exception:  # noqa: BLE001
        wrap = {}
    title = episode_title or str(wrap.get("title") or "untitled conversation")
    summary = str(wrap.get("summary") or "(no summary — the wrap-up call failed)")
    return save_episode(title, summary, msgs, eid=episode_id)


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
    header = working_header()
    episode_id, episode_title = header.get("episode"), header.get("title")
    resumed = working_messages()
    if resumed:
        where = (f"episode [{episode_id}] {episode_title}" if episode_id
                 else f"'{episode_title}'" if episode_title else "an untitled conversation")
        say(f"[dim](resuming {where} — {len(resumed)} messages in the working buffer)[/dim]")
    if list_episodes():
        say(f"[dim]({len(list_episodes())} episodes on the shelf — /episodes to browse)[/dim]")
    history = [{"role": "system", "content": system_msg(profile, episode_title)}] + resumed
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
        if user.lower() == "/episodes":
            eps = list_episodes()
            if not eps:
                say("[dim](no episodes yet — they're filed when you switch topics)[/dim]")
            for ep in eps:
                say(f"[dim][{ep['id']}] {ep['title']} · {len(ep['messages'])} msgs · {ep['summary']}[/dim]")
            continue
        if user.lower().startswith("/switch"):
            query = user[len("/switch"):].strip()
            target = find_episode(query) if query else None
            if query and not target:
                say("[dim](no episode matches that — /episodes to see the shelf)[/dim]")
                continue
            if target and target["id"] == episode_id:
                say("[dim](that's the episode you're already in)[/dim]")
                continue
            filed = file_working_episode(cli, episode_id, episode_title)
            if filed:
                say(f"[dim](filed [{filed['id']}] {filed['title']})[/dim]")
            if target:
                episode_id, episode_title = target["id"], target["title"]
                working_clear(episode_id, episode_title, target["messages"])
                history = ([{"role": "system", "content": system_msg(profile, episode_title)}]
                           + target["messages"])
                say(f"[dim](resumed [{episode_id}] {episode_title} — {target['summary']})[/dim]")
            else:
                episode_id = episode_title = None
                working_clear()
                history = [{"role": "system", "content": system_msg(profile)}]
                say("[dim](fresh episode — what shall we talk about?)[/dim]")
            continue
        if user.lower() == "/reset":
            working_clear()
            episode_id = episode_title = None
            history = [{"role": "system", "content": system_msg(profile)}]
            say("[dim](conversation + working buffer cleared — episodes and profile survive)[/dim]")
            continue
        if user.lower() == "/forget":
            forget()
            profile = recall()
            history[0] = {"role": "system", "content": system_msg(profile, episode_title)}
            say("[dim](profile deleted — we've never met)[/dim]")
            continue

        # episodic memory: from the 2nd message on, does this continue the theme?
        if len(history) > 2:
            det = detect_topic(cli, episode_title, user)
            if det and not det.get("same_episode", True):
                match = find_episode(str(det.get("matches_episode") or ""))
                if match and match["id"] == episode_id:
                    match = None
                offer = (f"resume [{match['id']}] {match['title']}" if match
                         else "start a new episode")
                ans = input(f"    (new topic: {det.get('topic', '?')} — file this chat "
                            f"and {offer}? [y/N]) > ").strip().lower()
                if ans.startswith("y"):
                    filed = file_working_episode(cli, episode_id, episode_title)
                    if filed:
                        say(f"[dim](filed [{filed['id']}] {filed['title']})[/dim]")
                    if match:
                        episode_id, episode_title = match["id"], match["title"]
                        working_clear(episode_id, episode_title, match["messages"])
                        history = ([{"role": "system", "content": system_msg(profile, episode_title)}]
                                   + match["messages"])
                        say(f"[dim](resumed [{episode_id}] {episode_title} — {match['summary']})[/dim]")
                    else:
                        episode_id, episode_title = None, str(det.get("topic") or "") or None
                        working_clear(None, episode_title)
                        history = [{"role": "system", "content": system_msg(profile, episode_title)}]

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
        working_append("user", user)        # the buffer only holds turns that
        working_append("assistant", reply)  # succeeded — a failed call was popped
        new = extract_profile(cli, profile, user)
        if new:
            profile = remember(new)
            history[0] = {"role": "system", "content": system_msg(profile, episode_title)}
            say(f"[dim](profile: {as_prompt(new)})[/dim]")

    meter.show()
    say("[bold green]bye![/bold green]")


if __name__ == "__main__":
    main()
