# NER MODEL FOR HISTORICAL ANALYSIS
## Team

| Name | GitHub | Role |
| Holovin Maksym |  |  |
| Anton Deputat |  |  |

## Problem and motivation

Lviv in the 15th–17th centuries council records mention thousands of people, and linking those mentions across documents helps historians study the connections inside the city's population. So we need to find which existing NER model finds person names in these manuscripts best.

## Data

Our primary dataset is a set of 15th–17th-century Lviv city council records provided by the Center for Digital Humanities, transcribed page by page while researchers process new scans.

| Set | Pages | Person mentions | Used for |
| Training data | 62 | 739 | Fine-tuning (`model_training/`) |
| Benchmark test set | 16 | 181 | Results and benchmarks |

Gold names are marked in the text as `<per>NAME<per>`.

## Models compared

7 original models (as published):

| Model | Type | Notes |
|-------|------|-------|
| `medieval-data/gliner_multi-v2.1-medieval-latin` | GLiNER | Base medieval Latin model |
| `magistermilitum/roberta-multilingual-medieval-ner` | XLM-R token classifier | |
| `efontes/efontes-ner` | XLM-R token classifier | Polish label names |
| `latincy/la_core_web_lg` | spaCy CNN | General-purpose Latin, not charter specific |
| `eyldlv/latin-ner` | spaCy transformer | Needs an old spaCy / transformers environment |
| `antondep2/efontes-ner-person` | XLM-R | Fine-tuned on our documents |
| `antondep2/gliner_lviv` | GLiNER | Fine-tuned on our documents |

6 modified variants (same weights, different input or decoding):

| Variant | What changes |
|---------|--------------|
| `antondep2`: 250-word chunks | Fixed 250-word windows instead of one sentence input |
| `antondep2`: 250-word chunks + first-subword decode | Entity span taken from first subword predictions |
| `antondep2`: 250-word chunks + first-subword + B-PER splits | Also starts a new name at every B-PER tag |
| `efontes-ner`: 250-word chunks | 250-word windows |
| `magistermilitum`: 250-word chunks | 250-word windows |
| `gliner_lviv`: 250-word chunks | 250-word windows |

7 hybrids (ensembles of already computed predictions):

| Ensemble | Rule |
|----------|------|
| GLiNER + efontes-ner | union |
| GLiNER + efontes-ner | vote ≥ 2 (both agree) |
| all 7 original models | vote ≥ 2 |
| all 7 original models | union |
| `antondep2` + `gliner_lviv` | union |
| `antondep2` + `gliner_lviv` | vote ≥ 2 (both agree) |
| `antondep2` + `gliner_lviv` + `efontes` chunks | majority, 2 of 3 |

In total that is 20 prediction sets. Post-processing rules are applied to the stored predictions of each single model.They are scored separately but are not counted as extra models.


## Approach

1. **Predict.** Each model reads a page and returns person name strings.
2. **Clean.** Titles and formulas that models glue onto names (*honesta*, *famatus*, *civis*…) are trimmed from span edges. The word list is compiled by hand from typical errors (`tuning.py`).
3. **Score.** Predictions are compared to the gold names with exact matching and partial matching. We report precision, recall and F1 and time per page.
4. **Tune.** Variants: 250-word chunks, first-subword decoding, saint-name filtering, snapping to word boundaries.
5. **Ensemble.** Union and majority vote over several models.
6. **Report.** An HTML page underlines gold names and each models findings on every page.

## Results

16 test pages, 181 gold mentions, Colab GPU. F1 is partial-match unless stated.

### Original models

| Model | Partial P / R / F1 | Exact P / R / F1 | ms / page |
|-------|--------------------|------------------|-----------|
| `gliner_multi-v2.1-medieval-latin` | 0.98 / 0.64 / 0.77 | 0.73 / 0.47 / 0.57 | 2789 |
| `magistermilitum` RoBERTa | 0.64 / 0.88 / 0.74 | 0.26 / 0.35 / 0.30 | 821 |
| `antondep2/gliner_lviv` | 0.96 / 0.84 / **0.90** | 0.81 / 0.71 / **0.76** | 2367 |
| `efontes/efontes-ner` | 0.74 / 0.93 / 0.82 | 0.44 / 0.55 / 0.49 | 953 |
| `antondep2/efontes-ner-person` | 0.92 / 0.68 / 0.78 | 0.41 / 0.30 / 0.34 | 739 |
| `latincy/la_core_web_lg` | 0.35 / 0.70 / 0.46 | 0.12 / 0.24 / 0.16 | 280 |
| `eyldlv/latin-ner` | 0.93 / 0.22 / 0.36 | 0.26 / 0.06 / 0.10 | 1393 |

### Modified variants

| Variant | Partial P / R / F1 | Exact P / R / F1 | ms / page |
|---------|--------------------|------------------|-----------|
| `antondep2`: 250-word chunks | 0.91 / 0.90 / 0.90 | 0.63 / 0.62 / 0.63 | 145 |
| `antondep2`: + first-subword decode | 0.93 / 0.89 / 0.91 | 0.70 / 0.67 / 0.68 | 146 |
| `antondep2`: + first-subword + B-PER splits | 0.92 / 0.93 / **0.93** | 0.73 / 0.74 / **0.74** | 146 |
| `efontes-ner`: 250-word chunks | 0.82 / 0.86 / 0.84 | 0.44 / 0.46 / 0.45 | 153 |
| `magistermilitum`: 250-word chunks | 0.74 / 0.88 / 0.81 | 0.31 / 0.36 / 0.33 | 151 |
| `gliner_lviv`: 250-word chunks | 0.96 / 0.83 / 0.89 | 0.80 / 0.69 / 0.74 | 3079 |

### Hybrids (ensembles)

| Ensemble | Partial P / R / F1 | Exact P / R / F1 |
|----------|--------------------|------------------|
| union: GLiNER + efontes-ner | 0.66 / 0.92 / 0.77 | 0.44 / 0.61 / 0.51 |
| vote ≥ 2: GLiNER + efontes-ner | 1.00 / 0.50 / 0.67 | 0.70 / 0.35 / 0.47 |
| vote ≥ 2: all 7 original models | 0.74 / 0.90 / 0.81 | 0.43 / 0.52 / 0.47 |
| union: all 7 original models | 0.25 / 0.99 / 0.40 | 0.20 / 0.77 / 0.31 |
| union: `antondep2` + `gliner_lviv` | 0.83 / 0.88 / 0.86 | 0.70 / 0.75 / 0.73 |
| vote ≥ 2: `antondep2` + `gliner_lviv` | 0.97 / 0.72 / 0.83 | 0.78 / 0.58 / 0.66 |
| majority 2 of 3: `antondep2` + `gliner_lviv` + `efontes` chunks | 0.93 / 0.78 / 0.85 | 0.58 / 0.49 / 0.53 |

### Post-processing

Two rules were tested on the stored predictions - dropping saint names and extending mid word spans to whole words. Effects are small:

- **Snap** helps exact match most for models with ragged boundaries: `magistermilitum` 0.30 - 0.41, `efontes` chunks 0.45 - 0.52, `antondep2` base 0.34 - 0.46.
- **Saints** gives +0.01–0.02 partial F1 for `efontes-ner`.
- Neither rule changes the best `antondep2` variant or either `gliner` model.


## Quick start

```bash
git clone <repo-url> && cd ner_history_analysis
pip install -r requirements.txt
cd benchmark
# place new_test.zip next to run.py
python run.py
```

In Colab:

```python
!pip install -q transformers torch pandas tqdm gliner spacy spacy-transformers huggingface_hub
%cd ner_history_analysis/benchmark
!python run.py
```

`run.py` loads the data, benchmarks every model, evaluates post-processing and ensembles, prints a summary table and writes `highlight_report.html`. `eyldlv/latin-ner` needs spaCy 3.3.1 and an old transformers. It may fail to load in your environment.

## Repository structure

```
.
├── benchmark/
│   ├── run.py                 # entry point
│   ├── data.py                # loading and normalising the annotated pages
│   ├── metrics.py             # exact / partial matching, P / R / F1
│   ├── benchmark.py           # runs every model over every page and times it
│   ├── registry.py            # all models and variants in one place
│   ├── tuning.py              # span cleaning, chunking, subword decoding, post-processing
│   ├── ensembles.py           # union and vote ensembles
│   ├── report.py              # builds the HTML highlight report
│   ├── report_template.html   # page template filled by report.py
│   └── models/                # one loader per model
├── model_training/            # fine-tuning notebooks
├── docs/
├── requirements.txt
└── README.md
```


## NER USED:
medieval-data/gliner_multi-v2.1-medieval-latin <br>
magistermilitum/roberta-multilingual-medieval-ner <br>
efontes/efontes-ner <br>
latincy/la_core_web_lg <br>
eyldlv/latin-ner   (the GREATESET OF ALL TIME) <br>
