# Length-Controlled Abstractive Summarization (BART on CNN/DailyMail)

This repository contains a reproducible **baseline** and a **length-controlled** extension for abstractive summarization using BART.
The controllable variant supports length buckets via input control tokens: `<SHORT>`, `<MEDIUM>`, `<LONG>`.

## Quick Start

```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate       # Linux/Mac
# or: py -m venv .venv && .venv\Scripts\activate       # Windows

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Login to HF if needed
# huggingface-cli login

# 4) Baseline training
python -m src.train_baseline   --model_name facebook/bart-large-cnn   --output_dir results/baseline   --epochs 1 --lr 3e-5 --per_device_train_batch_size 4   --max_src_len 1024 --max_tgt_len 128

# 5) Controlled training (with length tokens)
python -m src.train_controlled   --model_name facebook/bart-large-cnn   --output_dir results/controlled   --epochs 1 --lr 3e-5 --per_device_train_batch_size 4   --max_src_len 1024 --max_tgt_len 128   --short_thr 60 --medium_thr 100

# 6) Evaluation (ROUGE + length accuracy; BARTScore optional)
python -m src.evaluate   --model_dir results/controlled   --split validation   --max_src_len 1024 --max_tgt_len 128   --control_token MEDIUM   --short_thr 60 --medium_thr 100   --compute_bartscore false   --save_csv true --save_charts true
```

## Length Buckets
- **SHORT**: `len(summary_words) < short_thr`  
- **MEDIUM**: `short_thr ≤ len < medium_thr`  
- **LONG**: `len ≥ medium_thr`  

Default thresholds: `short_thr=60`, `medium_thr=100`.

## Configs
You can also run with provided configs:
```bash
python -m src.train_baseline --config config/baseline.json
python -m src.train_controlled --config config/controlled.json
```

## BARTScore (Optional)
We include an optional BARTScore evaluation. To enable:
```bash
pip install bart-score
python -m src.evaluate --model_dir results/controlled --compute_bartscore true
```
> Note: If installation fails, check the original repo for alternative install instructions.

## Reproducibility
- Seed is fixed to `42` unless overridden via `--seed`.
- We log key hyperparameters and ROUGE metrics to stdout and save to `results/**/metrics.json` and `eval_metrics.json`.
- Per-example metrics are saved to `predictions.jsonl` (and `predictions.csv`) for significance testing.

## Dataset
- Hugging Face: `cnn_dailymail` with configuration `3.0.0`.
- Automatically downloaded by the scripts.

## Significance Tests
Compare two runs (e.g., baseline vs controlled) with paired tests and bootstrap:
```bash
python -m src.significance   --file_a results/baseline/predictions.jsonl   --file_b results/controlled/predictions.jsonl   --metric rougeL_f1 --n_bootstrap 1000
```

## Charts & CSV Outputs
The evaluation script saves per-example metrics to JSONL/CSV and generates PNG charts.

Example:
```bash
python -m src.evaluate   --model_dir results/controlled   --split validation   --max_src_len 1024 --max_tgt_len 128   --control_token MEDIUM   --short_thr 60 --medium_thr 100   --compute_bartscore false   --save_csv true --save_charts true
```
Outputs:
- `eval_metrics.json`, `predictions.jsonl`, `predictions.csv`, `metrics_summary.csv`
- Figures in `results/**/figures/`: length histograms, ROUGE distributions, overall ROUGE bar chart, and (optional) BARTScore histogram.

## Poster Examples (Side-by-side SHORT/MEDIUM/LONG)
```bash
python -m src.make_poster_examples   --model_dir results/controlled   --split validation   --num_examples 3   --max_src_len 1024 --max_tgt_len 128
```
Outputs:
- `results/controlled/poster_examples.csv`
- `results/controlled/poster_examples.md`
