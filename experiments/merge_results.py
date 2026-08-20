"""Merge benchmark and ten-seed FedKAPT results into the figure-ready JSON."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_rel

import plot_main_results as plots

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "models" / "benchmark_raw" / "audit_results.json"
CORE = ROOT / "results" / "models" / "main_method_10seed.json"
OUT = ROOT / "results" / "models" / "audit_results.json"
METHOD = "FedKAPT (Ours)"
METRICS = ("accuracy", "macro_f1", "macro_auc")


def summary(runs):
    out = {}
    ours = {m: np.asarray([r[METHOD][m] for r in runs]) for m in METRICS}
    for method in plots.METHODS:
        out[method] = {}
        for metric in METRICS:
            vals = np.asarray([r[method][metric] for r in runs], float)
            half = t.ppf(.975, len(vals)-1) * vals.std(ddof=1) / math.sqrt(len(vals))
            item = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                    "ci95": [float(vals.mean()-half), float(vals.mean()+half)],
                    "values": vals.tolist()}
            if method != METHOD:
                item["paired_t_vs_fedkapt_p"] = float(
                    ttest_rel(vals, ours[metric]).pvalue)
                item["fedkapt_win_rate"] = float(np.mean(ours[metric] > vals))
            out[method][metric] = item
    return out


def main():
    audit = json.loads(BASE.read_text())
    core = json.loads(CORE.read_text())
    seeds = core["protocol"]["seeds"]
    for ds in ("cwru", "wdbc"):
        runs = audit["datasets"][ds]["runs"]
        for idx, seed in enumerate(seeds):
            runs[idx][METHOD] = core["runs"][ds][str(seed)]
        audit["datasets"][ds]["summary"] = summary(runs)
    audit["protocol"]["primary_method"] = "FedKAPT"
    OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    plots.configure()
    plots.FIGDIR.mkdir(parents=True, exist_ok=True)
    plots.save_main(audit)
    print(OUT.resolve())
    for ds in ("cwru", "wdbc"):
        row = audit["datasets"][ds]["summary"][METHOD]
        print(ds, *(f"{100*row[m]['mean']:.2f}+-{100*row[m]['std']:.2f}"
                    for m in METRICS))


if __name__ == "__main__":
    main()
