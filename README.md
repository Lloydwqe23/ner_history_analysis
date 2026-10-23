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

## Quick start

```bash
git clone <repo-url> && cd ner_history_analysis
pip install -r requirements.txt
cd benchmark
# place new_test.zip next to run.py (data is not in git)
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
