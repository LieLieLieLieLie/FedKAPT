"""Relational-only (rho=0) WDBC stress test for the KBS manuscript.

The target-test labels are used only after training to report assignment and
prediction metrics.  Alignment, screening, and model fitting consume the DP
source prototype release and unlabeled target-training records only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from component_study import DEFAULT_OUT, SEEDS, apply_variant, args_for
from identifiability_study import reliability_records
from main import build_config
from trainer import FedKAPTTrainer


def run(out_dir: Path, seeds, perturbations: int):
    rows = []
    for seed in seeds:
        print(f"RELATIONAL-ONLY wdbc seed={seed}", flush=True)
        ns = args_for(seed, "wdbc", out_dir)
        ns.exp_name = f"relational_only_wdbc_s{seed}"
        cfg = build_config(ns, "wdbc")
        apply_variant(cfg, "FedKAPT")
        cfg.privacy.release_dim = 10
        cfg.ot.clean_alignment = False
        cfg.ot.direct_cost_weight = 0.0

        trainer = FedKAPTTrainer(cfg)
        trainer.phase1_prototypes()
        trainer.phase2_ot_alignment()
        rel = reliability_records(trainer, [perturbations])[0]
        gen_tr, var_tr, cond_tr, pseudo, gen_te = trainer.phase3_generate()
        weights = trainer.phase4_filter(gen_tr, var_tr, cond_tr, pseudo)
        trainer.phase5_train_classifiers(gen_tr, gen_te, pseudo, weights)
        metrics = trainer.phase5_evaluate()["FedKAPT"]
        rows.append({
            "seed": int(seed),
            "assignment_accuracy": float(rel["assignment_accuracy"]),
            "accepted": bool(rel["accepted"]),
            "exact_assignment_stability": float(
                rel["exact_assignment_stability"]),
            "stability_radius": float(rel["stability_radius"]),
            "eta95": float(rel["eta95"]),
            "metrics": metrics,
        })

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    payload = {
        "protocol": {
            "dataset": "wdbc",
            "seeds": [int(s) for s in seeds],
            "direct_cost_weight_rho": 0.0,
            "source_message": "DP-sanitized class prototypes only",
            "release_dimension": 10,
            "perturbations": int(perturbations),
            "selection_uses_target_test_labels": False,
        },
        "runs": rows,
        "summary": {
            "assignment_accuracy": mean("assignment_accuracy"),
            "accept_rate": float(np.mean([r["accepted"] for r in rows])),
            "exact_assignment_stability": mean(
                "exact_assignment_stability"),
            "accuracy": float(np.mean([r["metrics"]["accuracy"] for r in rows])),
            "macro_f1": float(np.mean([r["metrics"]["macro_f1"] for r in rows])),
            "macro_auc": float(np.mean([r["metrics"]["macro_auc"] for r in rows])),
        },
    }
    path = out_dir / "relational_only_study.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--perturbations", type=int, default=200)
    args = parser.parse_args()
    result = run(Path(args.out_dir), args.seeds, args.perturbations)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
