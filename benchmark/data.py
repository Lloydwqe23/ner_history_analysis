"""Reading the annotated pages: page text plus gold person names."""

import glob
import os
import re
import zipfile

GOLD_TAG_RE = re.compile(r"<per>(.*?)<per>", flags=re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
DATA_ZIP = "new_test.zip"
EXTRACT_DIR = "ner_test_data"


def normalize(s: str) -> str:
    """Collapse whitespace into single spaces."""
    return WHITESPACE_RE.sub(" ", s).strip()


def load_page(path: str):
    """Return (text, gold names) for one annotated page."""
    raw = open(path, encoding="utf-8").read()
    gold = [normalize(g) for g in GOLD_TAG_RE.findall(raw)]
    gold = [g for g in gold if g]
    plain_text = raw.replace("<per>", "")
    return plain_text, gold


def ensure_data_unzipped(zip_path=DATA_ZIP, extract_dir=EXTRACT_DIR):
    """Unzip the data once, if it is not extracted yet."""
    if os.path.isdir(extract_dir) and glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True):
        return
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"{zip_path} not found in {os.getcwd()}. Upload it here (or edit "
            f"DATA_ZIP at the top of this file to point at the right path) "
            f"before running."
        )
    print(f"unzipping {zip_path} -> {extract_dir}/ ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)


def load_dataset(zip_path=DATA_ZIP, extract_dir=EXTRACT_DIR):
    """Load every annotated page as {"path", "text", "gold"}."""
    ensure_data_unzipped(zip_path, extract_dir)
    pages = []
    txt_files = sorted(glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True))
    for path in txt_files:
        plain_text, gold = load_page(path)
        if gold:
            pages.append({"path": path, "text": plain_text, "gold": gold})
    print(f"found {len(txt_files)} .txt files under {extract_dir}/, {len(pages)} of them annotated with <per> tags")
    return pages


def load_page_with_gold_spans(path: str):
    """Return (text, [(start, end), ...]) with character offsets of the gold names."""
    raw = open(path, encoding="utf-8").read()
    plain = raw.replace("<per>", "")
    spans, removed = [], 0
    for m in GOLD_TAG_RE.finditer(raw):
        s = m.start() - removed
        spans.append((s, s + len(m.group(1))))
        removed += 2 * len("<per>")
    return plain, spans