"""Scoring: exact and partial match, precision, recall, F1."""

import statistics
from dataclasses import dataclass, field


def is_partial(pred: str, gold: str) -> bool:
    p, g = pred.lower(), gold.lower()
    if not p or not g:
        return False
    return p in g or g in p


def match_page(preds: list[str], gold: list[str]):
    """Greedy one-to-one matching. Returns (exact_tp, partial_tp, false_positives)."""
    gold_left = list(gold)
    preds_left = list(preds)
    exact_tp = 0
    partial_tp = 0
    matched_pred_idx = set()

    for gi, g in enumerate(gold_left):
        for pi, p in enumerate(preds_left):
            if pi in matched_pred_idx:
                continue
            if p.lower() == g.lower():
                matched_pred_idx.add(pi)
                exact_tp += 1
                partial_tp += 1
                gold_left[gi] = None
                break

    for gi, g in enumerate(gold_left):
        if g is None:
            continue
        for pi, p in enumerate(preds_left):
            if pi in matched_pred_idx:
                continue
            if is_partial(p, g):
                matched_pred_idx.add(pi)
                partial_tp += 1
                break

    false_positives = [p for pi, p in enumerate(preds_left) if pi not in matched_pred_idx]
    return exact_tp, partial_tp, false_positives


@dataclass
class ModelResult:
    name: str
    total_preds: int = 0
    total_gold: int = 0
    exact_tp: int = 0
    partial_tp: int = 0
    false_positives: list = field(default_factory=list)
    page_times_ms: list = field(default_factory=list)
    page_raw_preds: list = field(default_factory=list)
    page_preds: list = field(default_factory=list)
    n_pages: int = 0
    load_time_s: float = 0.0
    n_params_m: float = None

    def precision_exact(self):
        return self.exact_tp / self.total_preds if self.total_preds else 0.0

    def recall_exact(self):
        return self.exact_tp / self.total_gold if self.total_gold else 0.0

    def f1_exact(self):
        p, r = self.precision_exact(), self.recall_exact()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def precision_partial(self):
        return self.partial_tp / self.total_preds if self.total_preds else 0.0

    def recall_partial(self):
        return self.partial_tp / self.total_gold if self.total_gold else 0.0

    def f1_partial(self):
        p, r = self.precision_partial(), self.recall_partial()
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def avg_ms_per_page(self):
        return statistics.mean(self.page_times_ms) if self.page_times_ms else 0.0

    def pages_per_sec(self):
        avg_s = self.avg_ms_per_page() / 1000
        return 1 / avg_s if avg_s else 0.0


def print_model_report(res: ModelResult):
    print(f"\n{'=' * 20} {res.name} {'=' * 20}")
    print(f"Pages evaluated: {res.n_pages} | Total Gold: {res.total_gold} | Total Preds: {res.total_preds}")
    fp_preview = res.false_positives[:15]
    print(f"False Positives (first 15): {fp_preview}")
    print(f"EXACT MATCH   -> Precision: {res.precision_exact():.2f} | Recall: {res.recall_exact():.2f} | F1: {res.f1_exact():.2f}")
    print(f"PARTIAL MATCH -> Precision: {res.precision_partial():.2f} | Recall: {res.recall_partial():.2f} | F1: {res.f1_partial():.2f}")
    print(f"Compute: {res.avg_ms_per_page():.1f} ms/page | {res.pages_per_sec():.2f} pages/sec | load time {res.load_time_s:.1f}s")


def print_summary_table(results):
    print("\n\n" + "=" * 100)
    print(f"{'Model':40s} {'F1(exact)':>10s} {'F1(partial)':>12s} {'ms/page':>10s} {'pages/sec':>10s}")
    print("-" * 100)
    for name, res in results.items():
        print(f"{name:40s} {res.f1_exact():>10.2f} {res.f1_partial():>12.2f} {res.avg_ms_per_page():>10.1f} {res.pages_per_sec():>10.2f}")
    print("=" * 100)
    print(
        "Tip: pick the model that maximizes F1(partial) per ms/page for your "
        "'good accuracy vs. compute cost' goal - divide F1 by ms/page for a "
        "quick efficiency score if useful."
    )