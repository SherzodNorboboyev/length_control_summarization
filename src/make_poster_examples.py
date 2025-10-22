import os
import argparse
import random
import textwrap
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import csv

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation", choices=["train","validation","test"])
    ap.add_argument("--num_examples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_src_len", type=int, default=1024)
    ap.add_argument("--max_tgt_len", type=int, default=128)
    ap.add_argument("--output_prefix", type=str, default="poster_examples")
    ap.add_argument("--truncate_chars", type=int, default=450, help="truncate article for readability in poster")
    return ap.parse_args()

def generate_with_token(model, tokenizer, article, prefix, max_src_len, max_tgt_len, device):
    text = (prefix + " " if prefix else "") + article
    inputs = tokenizer(text, max_length=max_src_len, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_length=max_tgt_len)
    return tokenizer.decode(gen[0], skip_special_tokens=True)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset("cnn_dailymail", "3.0.0")[args.split]
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[:args.num_examples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)

    rows = []
    md_lines = ["# Poster Examples", "Generated summaries with control tokens.\n"]

    for i, idx in enumerate(idxs, start=1):
        ex = ds[int(idx)]
        article = ex["article"]
        ref = ex["highlights"]

        short = generate_with_token(model, tokenizer, article, "<SHORT>", args.max_src_len, args.max_tgt_len, device)
        medium = generate_with_token(model, tokenizer, article, "<MEDIUM>", args.max_src_len, args.max_tgt_len, device)
        long = generate_with_token(model, tokenizer, article, "<LONG>", args.max_src_len, args.max_tgt_len, device)

        trunc = article[:args.truncate_chars].replace("\n", " ").strip()
        rows.append({
            "example_id": i,
            "article_trunc": trunc,
            "reference": ref,
            "pred_short": short,
            "pred_medium": medium,
            "pred_long": long,
            "len_short": len(short.split()),
            "len_medium": len(medium.split()),
            "len_long": len(long.split())
        })

        md_lines.extend([
            f"## Example {i}",
            "",
            "**Article (truncated):**",
            "",
            textwrap.fill(trunc, width=100),
            "",
            "**Reference:**",
            "",
            textwrap.fill(ref, width=100),
            "",
            "**Predictions:**",
            "",
            f"- `<SHORT>` ({len(short.split())} words):\n\n" + textwrap.fill(short, width=100),
            "",
            f"- `<MEDIUM>` ({len(medium.split())} words):\n\n" + textwrap.fill(medium, width=100),
            "",
            f"- `<LONG>` ({len(long.split())} words):\n\n" + textwrap.fill(long, width=100),
            "",
            "---",
            ""
        ])

    out_dir = args.model_dir
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, f"{args.output_prefix}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        import csv as _csv
        writer = _csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Markdown
    md_path = os.path.join(out_dir, f"{args.output_prefix}.md")
    with open(md_path, "w") as fmd:
        fmd.write("\n".join(md_lines))

    print(f"Saved poster examples to:\n- {csv_path}\n- {md_path}")

if __name__ == "__main__":
    main()