"""Local request classification on a Hugging Face model — no API key, no tokens.

    from hf_classifier import classify_local
    classify_local("write me a python function")
    → {"category": "code generation", "confidence": 0.93}

This is ZERO-SHOT classification: an NLI (entailment) model scores the text
against each candidate label as a hypothesis — "This example is {label}." —
so you get a classifier over any categories you name, with no training run.
Swap the CATEGORIES list and you have a different classifier.

The model (~440MB) downloads once to ~/.cache/huggingface on first use, then
loads from disk. Everything runs on your machine: free, private, and it
doesn't touch the metered class key.

Standalone smoke test:

    python labs/hf_classifier.py "can you mock up a landing page?"
"""

import sys

# Short names are what we report; the model scores the DESCRIPTIONS — small
# zero-shot models need a full phrase to latch onto, bare nouns mislead them.
CATEGORIES = {
    "writing or generating software code": "code generation",
    "sales, pricing, or closing business deals": "sales",
    "visual, UX, or product design": "design",
    "casual personal chat, not about work": "non-work related",
}
MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

_pipe = None  # loaded once, on first call — importing this module stays instant


def classify_local(text):
    """Return {"category": ..., "confidence": ...} for one message."""
    global _pipe
    if _pipe is None:
        from transformers import pipeline  # deferred — the heavy import
        _pipe = pipeline("zero-shot-classification", model=MODEL)
    out = _pipe(text, list(CATEGORIES), hypothesis_template="This message is about {}.")
    return {"category": CATEGORIES[out["labels"][0]],
            "confidence": round(out["scores"][0], 3)}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "write a python function to parse a csv"
    print(f"{text!r} → {classify_local(text)}")
