# Length-Controlled Abstractive Summarization (BART on CNN/DailyMail)

This repository contains a reproducible **baseline** and a **length-controlled** abstractive summarization system using BART.  
The controllable variant supports input control tokens: `<SHORT>`, `<MEDIUM>`, `<LONG>`.

---

## 🚀 Quick Start

**Note:** These codes have been tested and verified to work successfully with **Python 3.11.13** version on Mac.

```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate  # macOS/Linux
# or:
py -m venv .venv && .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ⚠️ If `BARTScore` Cannot Be Installed

The official **`bart-score`** package is *not* available on PyPI, so  
`pip install bart-score` **will not work**.

If you want to use the original implementation from the **Neulab** repository,  
you can install it manually as a local editable package by following these steps:

```bash
# 1. Clone the official repository
git clone https://github.com/neulab/BARTScore.git
cd BARTScore

# 2. Create a minimal build configuration
echo "[build-system]" > pyproject.toml
echo "requires = ['setuptools>=42','wheel']" >> pyproject.toml
echo "build-backend = 'setuptools.build_meta'" >> pyproject.toml

# 3. Add a lightweight setup.py
cat <<EOF > setup.py
from setuptools import setup, find_packages
setup(
    name='bart-score',
    version='0.0.0',
    packages=find_packages(),
)
EOF

# 4. Install locally in editable mode
python -m pip install -e .

# 5. Verify installation
python - << 'PY'
from bart_score import BARTScorer
print("✅ BARTScore installed successfully!")
PY
```
---

## 🛠️ Notes
- Tested on macOS (Apple Silicon, CPU mode) Python 3.11.13.
  If you encounter MPS memory issues, disable GPU:
  ```bash
  export PYTORCH_MPS_DISABLE=1
  ```

---

## Quick Start (using your existing env)
```bash
# Baseline training
python -m src.train_baseline --config config/baseline.json

# Length-controlled training
python -m src.train_controlled --config config/controlled.json

# Evaluation (ROUGE + Length Accuracy + Local BARTScore + charts/CSV)
python -m src.evaluate   --model_dir results/controlled   --split validation   --max_src_len 1024 --max_tgt_len 128   --control_token MEDIUM   --short_thr 60 --medium_thr 100   --compute_local_bartscore true   --bartscore_condition source   --save_csv true --save_charts true
```

## Significance Tests
```bash
python -m src.significance   --file_a results/baseline/predictions.jsonl   --file_b results/controlled/predictions.jsonl   --metric rougeL_f1 --n_bootstrap 1000
```

----
## Zero-shot baseline (500 val; 384/64):
- **64 cap; 500 val:**
```bash
export PYTORCH_MPS_DISABLE=1
python -m src.evaluate --model_dir facebook/bart-base \
  --split validation --num_examples 500 \
  --max_src_len 384 --max_tgt_len 64 \
  --compute_local_bartscore true --bartscore_condition source \
  --save_csv true --save_charts true
```

- **128 cap; 300 val; thresholds 60/100:**
```bash
python -m src.evaluate --model_dir facebook/bart-base \
  --split validation --num_examples 300 \
  --max_src_len 384 --max_tgt_len 128 \
  --control_token MEDIUM --short_thr 60 --medium_thr 100 \
  --compute_local_bartscore true --bartscore_condition source \
  --save_csv true --save_charts true --output_prefix medium_len128
```

- **64 cap; 500 val; thresholds 20/40:**
```bash
python -m src.evaluate --model_dir facebook/bart-base \
  --split validation --num_examples 500 \
  --max_src_len 384 --max_tgt_len 64 \
  --control_token LONG --short_thr 20 --medium_thr 40 \
  --compute_local_bartscore true --bartscore_condition source \
  --save_csv true --save_charts true --output_prefix long_len64
```

----
## Per-target caps (500 val; thresholds 20/40):
- **SHORT cap 32**
```bash
python -m src.evaluate --model_dir facebook/bart-base \
  --split validation --num_examples 500 \
  --max_src_len 384 --max_tgt_len 32 \
  --control_token SHORT --short_thr 20 --medium_thr 40 \
  --compute_local_bartscore true --bartscore_condition source \
  --save_csv true --save_charts true --output_prefix short_len32
```

- **MEDIUM cap 64**
```bash
python -m src.evaluate --model_dir facebook/bart-base \
  --split validation --num_examples 500 \
  --max_src_len 384 --max_tgt_len 64 \
  --control_token MEDIUM --short_thr 20 --medium_thr 40 \
  --compute_local_bartscore true --bartscore_condition source \
  --save_csv true --save_charts true --output_prefix medium_len64
```

- **LONG cap 96**
```bash
python -m src.evaluate --model_dir facebook/bart-base \
  --split validation --num_examples 500 \
  --max_src_len 384 --max_tgt_len 96 \
  --control_token LONG --short_thr 20 --medium_thr 40 \
  --compute_local_bartscore true --bartscore_condition source \
  --save_csv true --save_charts true --output_prefix long_len96
```


----

## 📊 Evaluation Metrics
- **ROUGE-1/2/L** — lexical overlap quality
- **Local BARTScore** — semantic similarity (source-conditioned likelihood)
- **Length Accuracy (LAcc)** — how often generated summaries fall within requested buckets

---

## 🧩 Example Output
| Target | ROUGE-L | Length Accuracy | Local BARTScore |
|---------|----------|----------------|-----------------|
| `<SHORT>` | 23.35 | 0.51 | -2.50 |
| `<MEDIUM>` | 30.33 | 0.09 | -1.33 |
| `<LONG>` | 29.02 | 1.00 | -1.07 |


---

## 📚 Citation
If you use this project, please cite the following:

- Lewis et al. (2019) — *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation*  
- Nallapati et al. (2016) — *Abstractive Text Summarization using Sequence-to-Sequence RNNs and Beyond*  
- Yuan et al. (2021) — *BARTScore: Evaluating Generated Text as Text Generation*  

---

© 2025 MBZUAI — NLP701 Class Project by Sherzod Norboboev and Abdulla Almessabi.
