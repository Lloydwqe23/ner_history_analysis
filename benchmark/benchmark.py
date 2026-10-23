"""Run every model from registry.py over every page and collect the results."""

import time
import traceback

from metrics import ModelResult, match_page, print_model_report, print_summary_table
from registry import MODEL_FACTORIES
from tuning import clean_entity_span


def run_benchmark(pages, model_name_filter=None):
    """Load each model, predict every page, score it. Returns {name: ModelResult}."""
    results = {}
    for name, factory in MODEL_FACTORIES.items():
        if model_name_filter and name not in model_name_filter:
            continue
        print(f"\n=== Loading {name} ===")
        t0 = time.time()
        try:
            predict_fn = factory()
        except Exception as e:
            print(f"  !! failed to load: {e}")
            print("  --- last frames of the traceback ---")
            print("".join(traceback.format_exc().splitlines(keepends=True)[-14:]))
            continue
        load_time = time.time() - t0

        res = ModelResult(name=name, load_time_s=load_time)
        for page in pages:
            gold = page["gold"]
            t0 = time.time()
            try:
                preds = predict_fn(page["text"])
            except Exception as e:
                print(f"  !! inference failed on {page['path']}: {e}")
                res.page_preds.append([])
                res.page_raw_preds.append([])
                continue
            elapsed_ms = (time.time() - t0) * 1000
            res.page_raw_preds.append(list(preds))
            preds = [clean_entity_span(p) for p in preds]
            preds = [p for p in preds if p]
            res.page_times_ms.append(elapsed_ms)
            res.page_preds.append(preds)
            res.n_pages += 1
            res.total_preds += len(preds)
            res.total_gold += len(gold)

            exact_tp, partial_tp, fps = match_page(preds, gold)
            res.exact_tp += exact_tp
            res.partial_tp += partial_tp
            res.false_positives.extend(fps)

        results[name] = res
        print_model_report(res)
    print_summary_table(results)
    return results