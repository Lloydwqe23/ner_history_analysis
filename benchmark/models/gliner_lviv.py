"""antondep2/gliner_lviv -- GLiNER fine-tuned on Lviv documents."""

import json
import os

from models.gliner_medieval import make_gliner_medieval


def _local_copy_with_fixed_tokenizer_config(model_id: str) -> str:
    """Download the repo and drop a list-valued 'extra_special_tokens' that crashes the tokenizer."""
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(model_id)
    cfg_path = os.path.join(local_dir, "tokenizer_config.json")
    if not os.path.exists(cfg_path):
        return local_dir
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg.get("extra_special_tokens"), list):
        return local_dir
    fixed_dir = local_dir.rstrip("/") + "_fixed"
    os.makedirs(fixed_dir, exist_ok=True)
    for name in os.listdir(local_dir):
        dst = os.path.join(fixed_dir, name)
        if name != "tokenizer_config.json" and not os.path.exists(dst):
            os.symlink(os.path.join(local_dir, name), dst)
    cfg.pop("extra_special_tokens")
    with open(os.path.join(fixed_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("  repaired tokenizer_config.json")
    return fixed_dir


def make_gliner_lviv(chunk_words=None):
    return make_gliner_medieval(
        model_id="antondep2/gliner_lviv",
        chunk_words=chunk_words,
        resolve_path=_local_copy_with_fixed_tokenizer_config,
    )