from typing import Dict
from transformers import AutoTokenizer

def get_tokenizer(model_name: str = "facebook/bart-large-cnn"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def build_preprocess_fn(tokenizer, max_src_len: int, max_tgt_len: int):
    def _fn(batch: Dict):
        model_inputs = tokenizer(
            batch["article"],
            max_length=max_src_len,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["highlights"],
                max_length=max_tgt_len,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _fn

def build_preprocess_with_control(tokenizer, max_src_len: int, max_tgt_len: int, short_thr: int, medium_thr: int):
    def _bucket(summary: str):
        n_words = len(summary.split())
        if n_words < short_thr:
            return "<SHORT>"
        elif n_words < medium_thr:
            return "<MEDIUM>"
        return "<LONG>"

    def _fn(batch: Dict):
        controls = [_bucket(s) for s in batch["highlights"]]
        prefixed_articles = [f"{c} " + a for c, a in zip(controls, batch["article"])]
        model_inputs = tokenizer(
            prefixed_articles,
            max_length=max_src_len,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["highlights"],
                max_length=max_tgt_len,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _fn