"""efontes/efontes-ner -- XLM-R for medieval Latin NER (labels OSOBA = person)."""

from models.common import discover_person_label_filter, run_pipe_batched, split_sentences


def make_efontes_ner():
    import torch
    from transformers import pipeline
    pipe = pipeline(
        "token-classification",
        model="efontes/efontes-ner",
        aggregation_strategy=None,
        device=0 if torch.cuda.is_available() else -1,
    )
    state = {"is_person": None}

    def predict(text: str):
        sentences = split_sentences(text)
        if state["is_person"] is None:
            state["is_person"] = discover_person_label_filter(pipe, sentences[:5])
        return run_pipe_batched(pipe, sentences, state["is_person"])

    return predict