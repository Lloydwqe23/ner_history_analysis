"""Everything that gets benchmarked. Add a model here; comment a line out to skip it."""

from models.antondep2_ner import make_efontes_ner_person_finetuned
from models.efontes_ner import make_efontes_ner
from models.eyldlv import make_eyldlv_latin_ner
from models.gliner_lviv import make_gliner_lviv
from models.gliner_medieval import make_gliner_medieval
from models.latincy import make_latincy_lg
from models.magistermilitum import make_magistermilitum_roberta
from tuning import _cached_hf_pipe, make_antondep2_variant, make_chunked_variant

MODEL_FACTORIES = {
    "GLiNER Multi Medieval Latin": make_gliner_medieval,
    "magistermilitum RoBERTa": make_magistermilitum_roberta,
    "antondep2/gliner_lviv (GLiNER fine-tuned on Lviv docs)": make_gliner_lviv,
    "efontes/efontes-ner (new)": make_efontes_ner,
    "antondep2/efontes-ner-person (fine-tuned on Lviv docs)": make_efontes_ner_person_finetuned,
    "antondep2 variant: 250-word chunks (author's inference style)": make_antondep2_variant(250, "merge"),
    "antondep2 variant: 250-word chunks + first-subword decode": make_antondep2_variant(250, "first_subword"),
    "antondep2 variant: 250-word chunks + first-subword + B-PER splits": make_antondep2_variant(250, "first_subword", honor_b=True),
    "efontes-ner variant: 250-word chunks": make_chunked_variant(lambda: _cached_hf_pipe("efontes/efontes-ner"), 250, "merge"),
    "magistermilitum variant: 250-word chunks": make_chunked_variant(lambda: _cached_hf_pipe("magistermilitum/roberta-multilingual-medieval-ner"), 250, "merge"),
    "gliner_lviv variant: 250-word chunks": lambda: make_gliner_lviv(chunk_words=250),
    "latincy/la_core_web_lg (new)": make_latincy_lg,
    "eyldlv/latin-ner (spaCy, needs its OWN separate env)": make_eyldlv_latin_ner,
}