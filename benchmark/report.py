"""HTML report: gold names and what every model found, painted on the page text.

Colours: blue = gold, green = exact match, light green dashed = overlap with different
boundaries, red = false positive, yellow = missed gold name, orange = found by several
models but absent from gold.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

from data import load_page_with_gold_spans, normalize

HL_PRIORITY = {"missed": 1, "sus": 1, "fp": 2, "partial": 3, "exact": 4, "gold": 1}
HL_SUSPECT_MIN_MODELS = 3


def _load_template() -> str:
    return Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")


def model_category(name: str) -> str:
    """original = as published, modified = our "... variant: ...", hybrid = ensemble."""
    if name.startswith("Ensemble"):
        return "hybrid"
    if " variant: " in name:
        return "modified"
    return "original"


def print_model_inventory(results, ensembles):
    groups = {"original": [], "modified": [], "hybrid": []}
    for name in list(results) + list(ensembles):
        groups[model_category(name)].append(name)
    n_post = len(groups["original"]) + len(groups["modified"])
    print("\n" + "=" * 100)
    print("MODEL INVENTORY")
    for key, title in (("original", "original (as published, default inference)"),
                       ("modified", "modified (same weights, our input/decoding changes)"),
                       ("hybrid", "hybrid (ensembles of other models)")):
        print(f"  {len(groups[key]):2d}  {title}")
        for n in groups[key]:
            print(f"        - {n}")
    total = sum(len(v) for v in groups.values())
    print(f"  --  {total} prediction sets in total. On top of that, the post-processing A/B "
          f"(+saints / +snap / +both) re-scores each of the {n_post} single-model sets "
          f"3 more times, without being separate models.")
    print("=" * 100)


def _hl_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _hl_find(text, s):
    toks = s.split()
    if not toks:
        return []
    pat = r"\s+".join(re.escape(t) for t in toks)
    occ = [(m.start(), m.end()) for m in re.finditer(pat, text)]
    if not occ:
        occ = [(m.start(), m.end()) for m in re.finditer(pat, text, flags=re.IGNORECASE)]
    return occ


def _hl_place(text, preds, gold_spans):
    """Locate prediction strings in the text. Returns (spans, number not found)."""
    placed, unlocated = [], 0
    for s, n in Counter(preds).items():
        occ = _hl_find(text, s)
        if not occ:
            unlocated += n
            continue
        occ.sort(key=lambda se: (not any(_hl_overlap(se, g) for g in gold_spans), se[0]))
        placed.extend(occ[:n])
        unlocated += max(0, n - len(occ))
    return sorted(placed), unlocated


def _hl_classify(text, placed, gold_spans):
    """Label each span exact / partial / fp; also return the gold spans nothing touched."""
    covered = [False] * len(gold_spans)
    out = []
    for (s, e) in placed:
        cls = "fp"
        mine = normalize(text[s:e]).lower()
        for gi, g in enumerate(gold_spans):
            if _hl_overlap((s, e), g):
                covered[gi] = True
                if mine == normalize(text[g[0]:g[1]]).lower():
                    cls = "exact"
                elif cls != "exact":
                    cls = "partial"
        out.append((s, e, cls))
    missed = [(g[0], g[1], "missed") for gi, g in enumerate(gold_spans) if not covered[gi]]
    return out, missed


def _hl_flatten(n_chars, items):
    """Resolve overlaps by priority into non-overlapping [start, end, cls, title] segments."""
    cls = [None] * n_chars
    owner = [-1] * n_chars
    for idx, (s, e, c, _t) in enumerate(items):
        for i in range(s, min(e, n_chars)):
            if cls[i] is None or HL_PRIORITY[c] >= HL_PRIORITY[cls[i]]:
                cls[i], owner[i] = c, idx
    segs, i = [], 0
    while i < n_chars:
        if cls[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n_chars and cls[j + 1] == cls[i] and owner[j + 1] == owner[i]:
            j += 1
        segs.append([i, j + 1, cls[i], items[owner[i]][3]])
        i = j + 1
    return segs


def _hl_suspects(text, per_model_fp_spans):
    """Text runs that enough models mark as false positives, widened to whole words."""
    counts = [0] * len(text)
    for spans in per_model_fp_spans:
        mark = set()
        for s, e in spans:
            mark.update(range(s, e))
        for i in mark:
            counts[i] += 1
    items, i, n = [], 0, len(text)
    while i < n:
        if counts[i] < HL_SUSPECT_MIN_MODELS:
            i += 1
            continue
        j = i
        while j + 1 < n and counts[j + 1] >= HL_SUSPECT_MIN_MODELS:
            j += 1
        s, e = i, j + 1
        while s > 0 and text[s - 1].isalnum():
            s -= 1
        while e < n and text[e].isalnum():
            e += 1
        items.append((s, e, "sus", f"{max(counts[i:j + 1])} models found this, gold has no name here"))
        i = j + 1
    return items


def build_highlight_data(pages, results, ensembles):
    all_sets = {}
    all_sets.update({n: r for n, r in results.items() if len(r.page_preds) == len(pages)})
    all_sets.update({n: r for n, r in ensembles.items() if r is not None and len(r.page_preds) == len(pages)})

    models_meta = []
    for name, res in all_sets.items():
        models_meta.append({
            "name": name, "cat": model_category(name),
            "p_part": res.precision_partial(), "r_part": res.recall_partial(), "f1_part": res.f1_partial(),
            "p_ex": res.precision_exact(), "r_ex": res.recall_exact(), "f1_ex": res.f1_exact(),
            "ms": res.avg_ms_per_page(),
        })

    page_data, errors_per_page = [], []
    for pi, page in enumerate(pages):
        text, gold_spans = load_page_with_gold_spans(page["path"])
        entry = {"name": os.path.basename(page["path"]), "text": text, "n_gold": len(gold_spans),
                 "gold": _hl_flatten(len(text), [(s, e, "gold", "gold annotation") for s, e in gold_spans]),
                 "models": {}}
        fp_by_single_model, page_errors = [], 0
        for name, res in all_sets.items():
            placed, unlocated = _hl_place(text, res.page_preds[pi], gold_spans)
            classified, missed = _hl_classify(text, placed, gold_spans)
            items = [(s, e, c, c) for s, e, c in classified] + [(s, e, c, "missed") for s, e, c in missed]
            n_fp = sum(1 for _s, _e, c in classified if c == "fp")
            entry["models"][name] = {
                "segs": _hl_flatten(len(text), items),
                "exact": sum(1 for _s, _e, c in classified if c == "exact"),
                "partial": sum(1 for _s, _e, c in classified if c == "partial"),
                "fp": n_fp, "missed": len(missed), "n_preds": len(res.page_preds[pi]),
                "unlocated": unlocated,
            }
            if model_category(name) != "hybrid":
                fp_by_single_model.append([(s, e) for s, e, c in classified if c == "fp"])
                page_errors += n_fp + len(missed)
        entry["suspects"] = _hl_flatten(len(text), _hl_suspects(text, fp_by_single_model))
        page_data.append(entry)
        errors_per_page.append(page_errors)

    default_page = max(range(len(pages)), key=lambda i: errors_per_page[i]) if pages else 0
    return {"models": models_meta, "pages": page_data, "default_page": default_page,
            "suspect_min": HL_SUSPECT_MIN_MODELS}


def write_highlight_html(pages, results, ensembles, out_path="highlight_report.html"):
    """Write one self-contained HTML report (open it in a browser)."""
    data = build_highlight_data(pages, results, ensembles)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_load_template().replace("__DATA__", payload))
    print(f"\nwrote {out_path}  ({len(data['pages'])} pages x {len(data['models'])} prediction sets; "
          f"opens on {data['pages'][data['default_page']]['name']}, the page with the most mistakes)")
    return out_path