"""antondep2/efontes-ner-person -- fine-tune of efontes-ner on Lviv documents."""

from models.common import discover_person_label_filter, run_pipe_batched, split_sentences

MODEL_NAME = "antondep2/efontes-ner-person"
BASE_TOKENIZER_NAME = "efontes/efontes-ner"

_PIPE_CACHE = {}


def _load_tokenizer():
    """Repo's tokenizer config is broken: try its tokenizer.json, else use the base model's."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception:
        pass
    try:
        from huggingface_hub import hf_hub_download
        tok_file = hf_hub_download(MODEL_NAME, "tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_file, model_max_length=512,
            bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>",
            unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
        )
        print("  loaded tokenizer.json directly (config is broken)")
        return tokenizer
    except Exception as e:
        print(f"  own tokenizer unusable ({type(e).__name__}); using {BASE_TOKENIZER_NAME}'s")
        return AutoTokenizer.from_pretrained(BASE_TOKENIZER_NAME)


def _build_pipe():
    import torch
    from transformers import AutoModelForTokenClassification, pipeline
    return pipeline(
        "token-classification",
        model=AutoModelForTokenClassification.from_pretrained(MODEL_NAME),
        tokenizer=_load_tokenizer(),
        aggregation_strategy=None,
        device=0 if torch.cuda.is_available() else -1,
    )


def _antondep2_pipe():
    """Build once; shared by the baseline and the decoding variants."""
    if "pipe" not in _PIPE_CACHE:
        _PIPE_CACHE["pipe"] = _build_pipe()
    return _PIPE_CACHE["pipe"]


def make_efontes_ner_person_finetuned():
    pipe = _antondep2_pipe()
    state = {"is_person": None}

    def predict(text: str):
        sentences = split_sentences(text)
        if state["is_person"] is None:
            state["is_person"] = discover_person_label_filter(pipe, sentences[:5])
        return run_pipe_batched(pipe, sentences, state["is_person"])

    return predict