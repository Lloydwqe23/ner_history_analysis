"""Improvements that need no retraining: chunked input, first-subword decoding, span cleaning, post-processing."""

import re

from data import normalize
from metrics import ModelResult, match_page
from models.antondep2_ner import _antondep2_pipe
from models.common import bare_label, discover_person_label_filter, merge_char_spans

BOUNDARY_NOISE_WORDS = {
    "honesta", "honesti", "honestus", "honestae",
    "famatus", "famati", "famato", "famatum", "famata",
    "generosa", "generosus", "generosi", "gener",
    "provido", "prouido", "providus",
    "judicio", "iudicio", "judicii", "iudicii",
    "civis", "civitatis", "civita",
    "matrimony", "matrimonii", "matrimonium",
    "coniunx", "coniuges", "coniugum", "coniun",
    "bedello", "bedellus", "bedelli",
    "juris", "officio", "officii",
    "banito", "bannito", "bannitus",
    "sancti", "sancta", "sanctus",
    "pleni", "plenus",
    "alutary", "alutarius",
    "incola", "incolae",
    "leopolien", "leopoliensis", "leopolien,",
    "primarum", "nuptiarum", "modernos", "praesentem", "praefata",
}

PUNCT_STRIP = ".,;:()[]\"'`"

SAINT_WORD_RE = re.compile(r"\b(?:sanct|beat)\w*", flags=re.IGNORECASE)

_HF_PIPES = {}


def clean_entity_span(name: str) -> str:
    """Trim punctuation and legal-formula words from both ends of a span."""
    words = name.strip(PUNCT_STRIP + " ").split()
    changed = True
    while changed and words:
        changed = False
        if words[0].strip(PUNCT_STRIP).lower() in BOUNDARY_NOISE_WORDS:
            words = words[1:]
            changed = True
        if words and words[-1].strip(PUNCT_STRIP).lower() in BOUNDARY_NOISE_WORDS:
            words = words[:-1]
            changed = True
    cleaned = " ".join(w.strip(PUNCT_STRIP) for w in words).strip()
    return cleaned


def merge_first_subword(raw_tokens, text, is_person, honor_b=False):
    """Word-level decoding: a word takes the label of its first subword.

    honor_b=True starts a new entity at every B-PER word.
    """
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if not words:
        return []
    by_start = {}
    for tok in raw_tokens:
        by_start.setdefault(tok["start"], tok)
    spans = []
    cur = None
    prev_i = None
    for i, (ws, we) in enumerate(words):
        tok = by_start.get(ws)
        if tok is None and ws > 0 and text[ws - 1].isspace():
            tok = by_start.get(ws - 1)
        if tok is None or not is_person(bare_label(tok["entity"])):
            if cur is not None:
                spans.append(cur)
                cur = None
            continue
        starts_new = honor_b and str(tok["entity"]).upper().startswith("B-")
        if cur is not None and prev_i == i - 1 and not starts_new:
            cur = (cur[0], we)
        else:
            if cur is not None:
                spans.append(cur)
            cur = (ws, we)
        prev_i = i
    if cur is not None:
        spans.append(cur)
    return [text[a:b] for a, b in spans]


def _cached_hf_pipe(model_name):
    """One shared pipeline per checkpoint."""
    if model_name not in _HF_PIPES:
        import torch
        from transformers import pipeline
        _HF_PIPES[model_name] = pipeline(
            "token-classification", model=model_name, aggregation_strategy=None,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _HF_PIPES[model_name]


def make_chunked_variant(get_pipe, chunk_words=250, decode="merge", honor_b=False):
    """Feed a model fixed-size word chunks instead of sentences; decode with "merge" or "first_subword"."""
    def factory():
        pipe = get_pipe()
        state = {"is_person": None}

        def predict(text: str):
            words = text.split()
            chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
            if not chunks:
                return []
            if state["is_person"] is None:
                state["is_person"] = discover_person_label_filter(pipe, chunks[:2])
            raw_batches = pipe(chunks)
            if raw_batches and isinstance(raw_batches[0], dict):
                raw_batches = [raw_batches]
            out = []
            for chunk, raw in zip(chunks, raw_batches):
                if decode == "first_subword":
                    spans = merge_first_subword(raw, chunk, state["is_person"], honor_b=honor_b)
                else:
                    spans = merge_char_spans(raw, chunk, state["is_person"])
                out.extend(normalize(sp) for sp in spans)
            return out

        return predict
    return factory


def make_antondep2_variant(chunk_words=250, decode="merge", honor_b=False):
    return make_chunked_variant(_antondep2_pipe, chunk_words, decode, honor_b)


def drop_saint_spans(raw_preds):
    """Drop raw predictions that mention a saint. Run before clean_entity_span strips "Sancti"."""
    return [p for p in raw_preds if not SAINT_WORD_RE.search(p)]


def snap_to_word_boundaries(pred, text):
    """Grow a span that starts or ends mid-word to whole words, unless an occurrence is already word-bounded."""
    if not pred:
        return pred
    starts = [m.start() for m in re.finditer(re.escape(pred), text)]
    if not starts:
        return pred

    def bounded(s):
        e = s + len(pred)
        left_ok = s == 0 or not text[s - 1].isalnum()
        right_ok = e == len(text) or not text[e].isalnum()
        return left_ok and right_ok

    if any(bounded(s) for s in starts):
        return pred
    s = starts[0]
    e = s + len(pred)
    while s > 0 and text[s - 1].isalnum():
        s -= 1
    while e < len(text) and text[e].isalnum():
        e += 1
    return text[s:e]


def postprocess(raw_preds, text, saints=False, snap=False):
    if saints:
        raw_preds = drop_saint_spans(raw_preds)
    preds = [clean_entity_span(p) for p in raw_preds]
    preds = [p for p in preds if p]
    if snap:
        preds = [clean_entity_span(snap_to_word_boundaries(p, text)) for p in preds]
        preds = [p for p in preds if p]
    return preds


def evaluate_postprocessing(pages, results):
    """Re-score every model's raw predictions under each post-processing variant and print one table."""
    variants = [
        ("baseline", dict(saints=False, snap=False)),
        ("+saints", dict(saints=True, snap=False)),
        ("+snap", dict(saints=False, snap=True)),
        ("+both", dict(saints=True, snap=True)),
    ]
    print("\n\n" + "#" * 100)
    print("# POST-PROCESSING A/B (same stored predictions, no model reruns)")
    print("#   saints = drop 'Sanctae Annae'-style spans; snap = grow mid-word spans to whole words")
    print("#" * 100)
    print(f"{'Model':46s} {'variant':10s} {'P(part)':>8s} {'R(part)':>8s} {'F1(part)':>9s} {'F1(exact)':>10s}")
    print("-" * 100)
    for name, res in results.items():
        if len(res.page_raw_preds) != len(pages):
            continue
        for vname, kw in variants:
            r = ModelResult(name=name)
            for i, page in enumerate(pages):
                preds = postprocess(res.page_raw_preds[i], page["text"], **kw)
                exact_tp, partial_tp, _ = match_page(preds, page["gold"])
                r.total_preds += len(preds)
                r.total_gold += len(page["gold"])
                r.exact_tp += exact_tp
                r.partial_tp += partial_tp
            print(f"{name[:46]:46s} {vname:10s} {r.precision_partial():>8.2f} {r.recall_partial():>8.2f} "
                  f"{r.f1_partial():>9.2f} {r.f1_exact():>10.2f}")
        print("-" * 100)