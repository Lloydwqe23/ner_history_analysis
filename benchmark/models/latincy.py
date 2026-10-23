"""latincy/la_core_web_lg -- general-purpose Latin spaCy pipeline (pip wheel from the Hub)."""

import importlib.util
import subprocess
import sys

from data import normalize
from models.common import decide_person_labels


def _install_wheel():
    from huggingface_hub import list_repo_files
    wheels = [f for f in list_repo_files("latincy/la_core_web_lg") if f.endswith(".whl")]
    if not wheels:
        raise RuntimeError("No .whl found in the latincy/la_core_web_lg repo")
    name = sorted(wheels)[-1]
    url = f"https://huggingface.co/latincy/la_core_web_lg/resolve/main/{name}"
    print(f"  installing {name} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"la_core_web_lg @ {url}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-1500:])
        raise RuntimeError(f"pip install of la_core_web_lg failed (exit {result.returncode})")


def make_latincy_lg():
    """Plain spaCy CNN pipeline; installs without conflicts."""
    if importlib.util.find_spec("la_core_web_lg") is None:
        _install_wheel()
    import spacy
    nlp = spacy.load("la_core_web_lg")
    state = {"chosen": None}

    def predict(text: str):
        doc = nlp(text)
        if state["chosen"] is None:
            state["chosen"] = decide_person_labels({ent.label_ for ent in doc.ents})
        if state["chosen"] is None:
            return [normalize(ent.text) for ent in doc.ents]
        return [normalize(ent.text) for ent in doc.ents if ent.label_ in state["chosen"]]

    return predict