"""Ten-seed benchmark study for the FedKAPT manuscript."""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import t, ttest_rel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_config
from trainer import FedKAPTTrainer
from experiments.baselines import run_baseline_comparison


METRICS = ("accuracy", "macro_f1", "macro_auc")


def _args(seed, dataset, epochs, out_dir):
    return SimpleNamespace(
        source="product", target="real_world", cwru_source=0, cwru_target=2,
        data_dir="./data/office_home", cwru_data_dir="./data/cwru",
        wdbc_data_dir="./data/wdbc",
        epsilon=10.0, delta=1e-5, max_norm=1.0,
        ot_mass=None, ot_reg=0.05, latent_dim=128,
        cvae_epochs=epochs, downstream_epochs=100,
        beta=1.0, ot_lambda=0.1,
        seed=seed, exp_name=f"benchmark_{dataset}_seed{seed}",
        log_dir=os.path.join(out_dir, "logs"),
        save_dir=os.path.join(out_dir, "checkpoints"),
        results_dir=os.path.join(out_dir, "results"),
    )


def _summarize(runs):
    methods = sorted({m for run in runs for m in run})
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in METRICS:
            vals = np.asarray([
                run.get(method, {}).get(metric, np.nan) for run in runs
            ], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            half = (float(t.ppf(0.975, len(vals) - 1)) * std / np.sqrt(len(vals))
                    if len(vals) > 1 else 0.0)
            summary[method][metric] = {
                "mean": float(vals.mean()), "std": std,
                "ci95": [float(vals.mean() - half), float(vals.mean() + half)],
                "values": vals.tolist(),
            }
    ours = summary.get("FedKAPT (Ours)", {})
    for method in methods:
        if method == "FedKAPT (Ours)":
            continue
        for metric in METRICS:
            if metric not in ours or metric not in summary.get(method, {}):
                continue
            x = np.asarray(ours[metric]["values"])
            y = np.asarray(summary[method][metric]["values"])
            if len(x) == len(y) and len(x) >= 2:
                stat = ttest_rel(x, y, nan_policy="omit")
                summary[method][metric]["paired_t_vs_fedkapt_p"] = float(stat.pvalue)
                summary[method][metric]["fedkapt_win_rate"] = float(np.mean(x > y))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cwru", "wdbc"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 2026, 3141, 2718, 8080, 9001, 10007])
    parser.add_argument("--cvae_epochs", type=int, default=50)
    parser.add_argument("--out_dir", default="results/models/benchmark_raw")
    ns = parser.parse_args()
    os.makedirs(ns.out_dir, exist_ok=True)
    payload = {"protocol": {
        "seeds": ns.seeds, "cvae_epochs": ns.cvae_epochs,
        "epsilon": 10.0, "delta": 1e-5,
    }, "datasets": {}}
    for dataset in ns.datasets:
        runs = []
        for seed in ns.seeds:
            print(f"AUDIT {dataset} seed={seed}", flush=True)
            args = _args(seed, dataset, ns.cvae_epochs, ns.out_dir)
            cfg = build_config(args, dataset)
            cfg.downstream.epochs = 100
            # The primary FedKAPT method is the deterministic transported-
            # condition pathway; generation is evaluated only as an optional
            # extension in the component study.
            cfg.cvae.disable_generation = True
            cfg.filter.min_keep_ratio = 1.0
            cfg.filter.recon_uncertainty_pct = 100.0
            cfg.filter.sem_uncertainty_pct = 100.0
            trainer = FedKAPTTrainer(cfg)
            fedkapt = trainer.train_test()
            run = run_baseline_comparison(trainer.data, cfg, fedkapt, logger=None)
            runs.append(run)
            checkpoint = os.path.join(ns.out_dir, f"runs_{dataset}.json")
            with open(checkpoint, "w", encoding="utf-8") as stream:
                json.dump(runs, stream, indent=2)
        payload["datasets"][dataset] = {
            "runs": runs, "summary": _summarize(runs)}
        with open(os.path.join(ns.out_dir, "audit_results.json"),
                  "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
    print(os.path.abspath(os.path.join(ns.out_dir, "audit_results.json")))


if __name__ == "__main__":
    main()
