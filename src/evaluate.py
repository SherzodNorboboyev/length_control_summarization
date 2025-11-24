import os
import json
import argparse
from typing import List
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load as load_metric
from rouge_score import rouge_scorer
from tqdm import tqdm
import csv
import jsonlines

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
    ap.add_argument("--compute_bartscore", type=str, default="false", choices=["true","false"], help="Use built-in BARTScore implementation")
    ap.add_argument("--bartscore_condition", type=str, default="source", choices=["source","reference"], help="Condition on source article or reference when computing BARTScore")
    ap.add_argument("--output_prefix", type=str, default="")
    ap.add_argument("--save_csv", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--save_charts", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--charts_dir", type=str, default=None, help="optional dir for charts; defaults to <model_dir>/figures")
    ap.add_argument("--output_dir", type=str, default=None, help="Where to write eval artifacts. Defaults to results/eval_<model_id> if model_dir is a hub ID.")

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

    preds, refs, per_ex = [], [], []

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
        articles = [prefix + ex for ex in batch["article"]]
        inputs = tokenizer(articles, max_length=args.max_src_len, truncation=True, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=args.max_tgt_len)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds.extend(decoded)
        refs.extend([ex for ex in batch["highlights"]])

    # Aggregate ROUGE
    rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    # rouge_res = {k: round(v.mid.fmeasure * 100, 2) for k, v in rouge_res.items()}
    norm_rouge = {}
    for k, v in rouge_res.items():
        try:
            val = float(v.mid.fmeasure)  # old style (Score object)
        except AttributeError:
            val = float(v)               # new style (float)
        norm_rouge[k] = round(val * 100, 2)
    rouge_res = norm_rouge

    # Per-example metrics
    for p, r in zip(preds, refs):
        sc = scorer.score(r, p)  # (reference, prediction)
        row = {
            "rouge1_f1": sc["rouge1"].fmeasure,
            "rouge2_f1": sc["rouge2"].fmeasure,
            "rougeL_f1": sc["rougeL"].fmeasure,
            "pred_len_words": len(p.split()),
            "ref_len_words": len(r.split()),
            "prediction": p,
            "reference": r
        }
        per_ex.append(row)

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

    # Optional BARTScore (no external dependency)
    if args.compute_bartscore.lower() == "true":
        from src.bartscore import LocalBARTScorer
        if args.bartscore_condition == "source":
            cond_sources = [ex["article"] for ex in dataset]
        else:
            cond_sources = refs
        lbs = LocalBARTScorer(device=device)
        lbs_scores = lbs.score(cond_sources, preds, batch_size=args.batch_size, max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
        out["bartscore_mean"] = float(np.mean(lbs_scores))
        for row, s in zip(per_ex, lbs_scores):
            row["bartscore"] = float(s)

    # Save outputs
    # Decide where to write outputs
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        if os.path.isdir(args.model_dir):
            # Local fine-tuned model directory -> write there
            out_dir = args.model_dir
        else:
            # Hub model ID -> write to a safe local path
            safe = args.model_dir.replace("/", "__")
            out_dir = os.path.join("results", f"eval_{safe}")
    os.makedirs(out_dir, exist_ok=True)

    prefix_name = (args.output_prefix + "_") if args.output_prefix else ""

    with open(os.path.join(out_dir, f"{prefix_name}eval_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    pred_path = os.path.join(out_dir, f"{prefix_name}predictions.jsonl")
    with jsonlines.open(pred_path, mode="w") as writer:
        for row in per_ex:
            writer.write(row)

    if args.save_csv.lower() == "true" and len(per_ex) > 0:
        csv_path = os.path.join(out_dir, f"{prefix_name}predictions.csv")
        keys = list(per_ex[0].keys())
        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=keys)
            writer.writeheader()
            for row in per_ex:
                writer.writerow(row)

        summary_csv = os.path.join(out_dir, f"{prefix_name}metrics_summary.csv")
        with open(summary_csv, "w", newline="") as fsum:
            w = csv.writer(fsum)
            w.writerow(["metric", "value"])
            for k, v in out["rouge"].items():
                w.writerow([k, v])
            if "length_accuracy" in out:
                w.writerow(["length_accuracy", out["length_accuracy"]])
            if "bartscore_mean" in out:
                w.writerow(["bartscore_mean", out["bartscore_mean"]])

    # Charts
    if args.save_charts.lower() == "true" and len(per_ex) > 0:
        charts_dir = args.charts_dir or os.path.join(out_dir, "figures")
        os.makedirs(charts_dir, exist_ok=True)
        try:
            import matplotlib.pyplot as plt

            # Predicted length histogram
            plt.figure()
            plt.hist([row["pred_len_words"] for row in per_ex], bins=30)
            plt.title("Predicted Summary Length (words)")
            plt.xlabel("Words")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, f"{prefix_name}pred_len_hist.png"))
            plt.close()

            # Reference length histogram
            plt.figure()
            plt.hist([row["ref_len_words"] for row in per_ex], bins=30)
            plt.title("Reference Summary Length (words)")
            plt.xlabel("Words")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, f"{prefix_name}ref_len_hist.png"))
            plt.close()

            # ROUGE distributions
            for key in ["rouge1_f1", "rouge2_f1", "rougeL_f1"]:
                plt.figure()
                plt.hist([row[key] for row in per_ex], bins=30)
                plt.title(f"{key} distribution")
                plt.xlabel("Score")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, f"{prefix_name}{key}_hist.png"))
                plt.close()

            # Overall ROUGE bar chart
            plt.figure()
            labels = list(out["rouge"].keys())
            values = [out["rouge"][k] for k in labels]
            x = range(len(labels))
            plt.bar(x, values)
            plt.xticks(ticks=x, labels=labels)
            plt.ylabel("F1 (%)")
            plt.title("Overall ROUGE")
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, f"{prefix_name}rouge_overall_bar.png"))
            plt.close()

            # BARTScore distribution if available
            if any("bartscore" in row for row in per_ex):
                vals = [row["bartscore"] for row in per_ex if "bartscore" in row]
                if len(vals) > 0:
                    plt.figure()
                    plt.hist(vals, bins=30)
                    plt.title("BARTScore (avg log-prob) distribution")
                    plt.xlabel("Score (nats per token)")
                    plt.ylabel("Count")
                    plt.tight_layout()
                    plt.savefig(os.path.join(charts_dir, f"{prefix_name}bartscore_hist.png"))
                    plt.close()

        except Exception as e:
            with open(os.path.join(out_dir, f"{prefix_name}chart_error.log"), "w") as ferr:
                ferr.write(str(e))

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()