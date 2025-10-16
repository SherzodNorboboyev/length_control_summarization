import argparse
import json
import numpy as np
from scipy import stats

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_a", type=str, required=True, help="JSONL predictions with per-example metrics (system A)")
    ap.add_argument("--file_b", type=str, required=True, help="JSONL predictions with per-example metrics (system B)")
    ap.add_argument("--metric", type=str, default="rougeL_f1", help="per-example metric key")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    return ap.parse_args()

def read_jsonl(path):
    import jsonlines
    vals = []
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            if args.metric in obj:
                vals.append(float(obj[args.metric]))
    return np.array(vals)

def bootstrap_pvalue(diffs, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    observed = np.mean(diffs)
    count = 0
    for _ in range(n_bootstrap):
        resample = rng.choice(diffs, size=n, replace=True)
        if abs(np.mean(resample)) >= abs(observed):
            count += 1
    return (count + 1) / (n_bootstrap + 1)

if __name__ == "__main__":
    args = parse_args()
    a = read_jsonl(args.file_a)
    b = read_jsonl(args.file_b)
    assert len(a) == len(b), "Files must have the same number of examples"
    diffs = b - a
    t_stat, p_ttest = stats.ttest_rel(b, a, nan_policy="omit")
    p_boot = bootstrap_pvalue(diffs, n_bootstrap=args.n_bootstrap)
    out = {
        "metric": args.metric,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff_b_minus_a": float(np.mean(diffs)),
        "paired_ttest_pvalue": float(p_ttest),
        "bootstrap_pvalue": float(p_boot),
        "n": int(len(a))
    }
    print(json.dumps(out, indent=2))