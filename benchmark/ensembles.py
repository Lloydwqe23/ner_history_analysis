"""Combine models: union / vote over predictions that were already computed."""

from metrics import ModelResult, match_page, print_model_report

ENSEMBLE_RESULTS = {}


def same_entity_cluster(a, b, min_len=5, min_overlap_ratio=0.45):
    """True if two spans look like the same mention (stricter than partial scoring, so "Leo" does not merge into "Leopoli")."""
    a_l, b_l = a.lower().strip(), b.lower().strip()
    if a_l == b_l:
        return True
    shorter, longer = (a_l, b_l) if len(a_l) <= len(b_l) else (b_l, a_l)
    if len(shorter) < min_len or shorter not in longer:
        return False
    return (len(shorter) / len(longer)) >= min_overlap_ratio


def combine_predictions(per_model_preds, mode="union", min_votes=2):
    """Combine one page's predictions from several models.

    union: everything, only exact duplicates removed.
    vote: keep spans found by at least min_votes distinct models.
    """
    if mode == "union":
        seen = {}
        for preds in per_model_preds:
            for p in preds:
                if not p:
                    continue
                key = p.lower().strip()
                if key not in seen or len(p) > len(seen[key]):
                    seen[key] = p
        return list(seen.values())

    elif mode == "vote":
        tagged = []
        for model_idx, preds in enumerate(per_model_preds):
            for p in preds:
                if p:
                    tagged.append((model_idx, p))

        if not tagged:
            return []

        clusters = []
        for model_idx, pred in tagged:
            placed = False
            for cluster in clusters:
                rep = cluster["rep"]
                if same_entity_cluster(pred, rep):
                    cluster["items"].append((model_idx, pred))
                    if len(pred) > len(rep):
                        cluster["rep"] = pred
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": pred, "items": [(model_idx, pred)]})

        kept = []
        for c in clusters:
            distinct_models = {m for m, _ in c["items"]}
            if len(distinct_models) >= min_votes:
                kept.append(c["rep"])
        return kept

    else:
        raise ValueError(f"unknown ensemble mode: {mode!r}")


def evaluate_ensemble(pages, results, model_names, mode="union", min_votes=2, label=None):
    """Score a combination of already-benchmarked models. Returns None if fewer than 2 are usable."""
    usable = []
    for name in model_names:
        res = results.get(name)
        if res is None:
            print(f"  !! ensemble: '{name}' not found in results, skipping")
            continue
        if len(res.page_preds) != len(pages):
            print(f"  !! ensemble: '{name}' has {len(res.page_preds)} page_preds but "
                  f"there are {len(pages)} pages (likely failed to load/infer) -- skipping")
            continue
        usable.append(name)

    if len(usable) < 2:
        print(f"  !! ensemble needs >= 2 usable models, got {len(usable)} ({usable}) -- skipping")
        return None

    label = label or f"Ensemble[{mode}]({' + '.join(usable)})"
    ens = ModelResult(name=label)
    for i, page in enumerate(pages):
        gold = page["gold"]
        per_model_preds = [results[name].page_preds[i] for name in usable]
        combined = combine_predictions(per_model_preds, mode=mode, min_votes=min_votes)

        ens.page_preds.append(combined)
        ens.n_pages += 1
        ens.total_preds += len(combined)
        ens.total_gold += len(gold)

        exact_tp, partial_tp, fps = match_page(combined, gold)
        ens.exact_tp += exact_tp
        ens.partial_tp += partial_tp
        ens.false_positives.extend(fps)

    ENSEMBLE_RESULTS[ens.name] = ens
    print_model_report(ens)
    return ens