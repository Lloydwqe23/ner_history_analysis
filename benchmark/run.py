"""Run the whole benchmark: all models, post-processing table, ensembles, HTML report."""

from benchmark import run_benchmark
from data import load_dataset
from ensembles import ENSEMBLE_RESULTS, evaluate_ensemble
from report import print_model_inventory, write_highlight_html
from tuning import evaluate_postprocessing


def main():
    pages = load_dataset()
    print(f"Loaded {len(pages)} annotated pages, {sum(len(p['gold']) for p in pages)} gold person mentions total.")

    results = run_benchmark(pages)
    evaluate_postprocessing(pages, results)

    print("\n\n" + "#" * 100)
    print("# ENSEMBLES (combining predictions already computed above, no reloading)")
    print("#" * 100)

    evaluate_ensemble(
        pages, results,
        ["GLiNER Multi Medieval Latin", "efontes/efontes-ner (new)"],
        mode="union",
        label="Ensemble[union]: GLiNER + efontes-ner",
    )
    evaluate_ensemble(
        pages, results,
        ["GLiNER Multi Medieval Latin", "efontes/efontes-ner (new)"],
        mode="vote", min_votes=2,
        label="Ensemble[vote>=2]: GLiNER + efontes-ner",
    )

    all_loaded = [name for name, res in results.items()
                  if len(res.page_preds) == len(pages) and " variant: " not in name]
    evaluate_ensemble(
        pages, results, all_loaded,
        mode="vote", min_votes=2,
        label=f"Ensemble[vote>=2]: all {len(all_loaded)} loaded models",
    )
    evaluate_ensemble(
        pages, results, all_loaded,
        mode="union",
        label=f"Ensemble[union]: all {len(all_loaded)} loaded models",
    )

    BEST_ANTONDEP2 = "antondep2 variant: 250-word chunks + first-subword + B-PER splits"
    GLINER_LVIV = "antondep2/gliner_lviv (GLiNER fine-tuned on Lviv docs)"
    EFONTES_CHUNKS = "efontes-ner variant: 250-word chunks"
    evaluate_ensemble(
        pages, results, [BEST_ANTONDEP2, GLINER_LVIV],
        mode="union",
        label="Ensemble[union]: best antondep2 + gliner_lviv",
    )
    evaluate_ensemble(
        pages, results, [BEST_ANTONDEP2, GLINER_LVIV],
        mode="vote", min_votes=2,
        label="Ensemble[vote>=2 = both agree]: best antondep2 + gliner_lviv",
    )
    evaluate_ensemble(
        pages, results, [BEST_ANTONDEP2, GLINER_LVIV, EFONTES_CHUNKS],
        mode="vote", min_votes=2,
        label="Ensemble[majority 2 of 3]: antondep2 best + gliner_lviv + efontes chunks",
    )

    print_model_inventory(results, ENSEMBLE_RESULTS)
    report_path = write_highlight_html(pages, results, ENSEMBLE_RESULTS, "highlight_report.html")
    try:
        from google.colab import files
        files.download(report_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()