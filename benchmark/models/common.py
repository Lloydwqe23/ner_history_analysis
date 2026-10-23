"""Shared helpers: sentence splitting and turning token predictions into name strings."""

import re

from data import normalize

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+|\r?\n")
PERSON_KEYWORDS = ("PER", "NAME", "OSOBA")


def split_sentences(text: str):
    """Split on periods, semicolons and newlines."""
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def chunk_by_word_budget(sentences, max_words=230):
    """Group consecutive sentences into chunks of up to max_words words."""
    chunks = []
    cur, cur_words = [], 0
    for s in sentences:
        w = len(s.split())
        if cur and cur_words + w > max_words:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(s)
        cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def bare_label(raw_label: str) -> str:
    """'B-PERS' -> 'PERS'."""
    if len(raw_label) > 2 and raw_label[1] == "-" and raw_label[0] in "BIES":
        return raw_label[2:]
    return raw_label


def merge_char_spans(raw_token_preds, text, is_person):
    """Merge adjacent person tokens into strings sliced from text by character offsets."""
    spans = []
    cur_start = cur_end = None
    for tok in raw_token_preds:
        if not is_person(bare_label(tok["entity"])):
            if cur_start is not None:
                spans.append((cur_start, cur_end))
                cur_start = cur_end = None
            continue
        if cur_start is None:
            cur_start, cur_end = tok["start"], tok["end"]
        elif tok["start"] <= cur_end + 1:
            cur_end = max(cur_end, tok["end"])
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = tok["start"], tok["end"]
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return [text[s:e] for s, e in spans]


def decide_person_labels(labels: set, keywords=PERSON_KEYWORDS):
    """Pick which observed labels mean PERSON; None if no labels seen yet."""
    if not labels:
        return None
    person_like = {l for l in labels if any(k in l.upper() for k in keywords)}
    chosen = person_like or labels
    print(f"  labels seen: {labels} -> PERSON = {chosen}")
    return chosen


def discover_person_label_filter(pipe, probe_sentences, keywords=PERSON_KEYWORDS):
    """Probe the pipe on a few sentences and return an is-person predicate."""
    seen = set()
    for s in probe_sentences:
        try:
            seen.update(bare_label(r["entity"]) for r in pipe(s))
        except Exception:
            continue
    chosen = decide_person_labels(seen, keywords)
    if chosen is None:
        print("  (!) probe found no entities; keeping all labels")
        return lambda label: True
    return lambda label: label in chosen


def run_pipe_batched(pipe, sentences, is_person):
    """Run a token-classification pipe on all sentences in one batched call."""
    if not sentences:
        return []
    raw_batches = pipe(sentences)
    if raw_batches and isinstance(raw_batches[0], dict):
        raw_batches = [raw_batches]
    preds = []
    for sent, raw in zip(sentences, raw_batches):
        preds.extend(normalize(s) for s in merge_char_spans(raw, sent, is_person))
    return preds