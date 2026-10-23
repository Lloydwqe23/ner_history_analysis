"""eyldlv/latin-ner -- spaCy pipeline; needs an old spaCy/transformers environment."""

from data import normalize
from models.common import decide_person_labels


def make_eyldlv_latin_ner():
    """spacy_transformers must be imported before spacy.load to register the 'transformer' factory."""
    import spacy
    import spacy_transformers
    from huggingface_hub import snapshot_download
    nlp = spacy.load(snapshot_download(repo_id="eyldlv/latin-ner"))
    state = {"chosen": None}

    def predict(text: str):
        doc = nlp(text)
        if state["chosen"] is None:
            state["chosen"] = decide_person_labels({ent.label_ for ent in doc.ents})
        if state["chosen"] is None:
            return [normalize(ent.text) for ent in doc.ents]
        return [normalize(ent.text) for ent in doc.ents if ent.label_ in state["chosen"]]

    return predict