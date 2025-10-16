import os
import json
import argparse
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load as load_metric
from rouge_score import rouge_scorer
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--max_src_len", type=int, default=1024)
    ap.add_argument("--max_tgt_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_examples", type=int, default=1000, help="limit for quick evaluation; -1 for full split")
    ap.add_argument("--control_token", type=str, default=None, choices=[None, "SHORT", "MEDIUM", "LONG"])
    ap.add_argument("--short_thr", type=int, default=60)
    ap.add_argument("--medium_thr", type=int, default=100)
    ap.add_argument("--compute_bartscore", type=str, default="false", choices=["true","false"])
    ap.add_argument("--output_prefix", type=str, default="")
    return ap.parse_args()

def length_bucket(n_words: int, short_thr: int, medium_thr: int) -> str:
    if n_words < short_thr:
        return "<SHORT>"
    elif n_words < medium_thr:
        return "<MEDIUM>"
    return "<LONG>"

def batch_iter(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset("cnn_dailymail", "3.0.0")[args.split]
    if 0 < args.num_examples < len(dataset):
        dataset = dataset.select(range(args.num_examples))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    rouge = load_metric("rouge")
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

    preds, refs, rows = [], [], []

    # Build (optional) control prefix
    prefix = ""
    if args.control_token is not None:
        if args.control_token == "SHORT":
            prefix = "<SHORT> "
        elif args.control_token == "MEDIUM":
            prefix = "<MEDIUM> "
        elif args.control_token == "LONG":
            prefix = "<LONG> "

    # Generate predictions
    for batch in tqdm(list(batch_iter(dataset, args.batch_size))):
        articles = [prefix + ex["article"] for ex in batch]
        inputs = tokenizer(articles, max_length=args.max_src_len, truncation=True, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=args.max_tgt_len)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds.extend(decoded)
        refs.extend([ex["highlights"] for ex in batch])

    # Aggregate ROUGE
    rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    rouge_res = {k: round(v.mid.fmeasure * 100, 2) for k, v in rouge_res.items()}

    # Per-example metrics
    per_ex = []
    for p, r in zip(preds, refs):
        sc = scorer.score(r, p)  # (reference, prediction)
        per_ex.append({
            "rouge1_f1": sc["rouge1"].fmeasure,
            "rouge2_f1": sc["rouge2"].fmeasure,
            "rougeL_f1": sc["rougeL"].fmeasure,
            "pred_len_words": len(p.split()),
            "ref_len_words": len(r.split()),
        })

    out = {"rouge": rouge_res}

    # Length accuracy if a control token is used
    if args.control_token is not None:
        desired = {"SHORT": "<SHORT>", "MEDIUM": "<MEDIUM>", "LONG": "<LONG>"}[args.control_token]
        hits = 0
        for row in per_ex:
            pred_bucket = length_bucket(row["pred_len_words"], args.short_thr, args.medium_thr)
            if pred_bucket == desired:
                hits += 1
            row["pred_bucket"] = pred_bucket
            row["desired_bucket"] = desired
        out["length_accuracy"] = round(hits / len(per_ex), 4)

    # Optional BARTScore
    if args.compute_bartscore.lower() == "true":
        try:
            from bart_score import BARTScorer
            scorer_bs = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
            bs_scores = scorer_bs.score(preds, refs, batch_size=args.batch_size)
            out["bartscore_mean"] = float(np.mean(bs_scores))
            # Add per-example BARTScore
            for row, s in zip(per_ex, bs_scores):
                row["bartscore"] = float(s)
        except Exception as e:
            out["bartscore_error"] = str(e)

    # Save outputs
    out_dir = args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    prefix = (args.output_prefix + "_") if args.output_prefix else ""
    with open(os.path.join(out_dir, f"{prefix}eval_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Save per-example predictions and metrics
    import jsonlines
    pred_path = os.path.join(out_dir, f"{prefix}predictions.jsonl")
    with jsonlines.open(pred_path, mode="w") as writer:
        for p, r, row in zip(preds, refs, per_ex):
            obj = {"prediction": p, "reference": r}
            obj.update(row)
            writer.write(obj)

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()