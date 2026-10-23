"""medieval-data/gliner_multi-v2.1-medieval-latin; also the generic GLiNER loader."""

from data import normalize
from models.common import chunk_by_word_budget, split_sentences


def make_gliner_medieval(threshold: float = 0.5,
                         model_id: str = "medieval-data/gliner_multi-v2.1-medieval-latin",
                         chunk_words=None,
                         resolve_path=None):
    """threshold stays at 0.5 (0.3 collapsed precision). chunk_words=None groups
    sentences into ~230-word chunks. resolve_path optionally maps model_id to a local folder."""
    from gliner import GLiNER
    model = GLiNER.from_pretrained(resolve_path(model_id) if resolve_path else model_id)
    labels = ["person"]

    def predict(text: str):
        if chunk_words:
            w = text.split()
            chunks = [" ".join(w[i:i + chunk_words]) for i in range(0, len(w), chunk_words)]
        else:
            chunks = chunk_by_word_budget(split_sentences(text))
        preds = []
        for chunk in chunks:
            ents = model.predict_entities(chunk, labels, threshold=threshold)
            preds.extend(normalize(e["text"]) for e in ents if e["label"] == "person")
        return preds

    return predict