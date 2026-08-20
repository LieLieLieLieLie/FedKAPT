"""Focused identifiability and condition-value study for the KBS manuscript.

The script addresses two reviewer-facing questions without using target-test
labels for model selection:

1. Can minimal disclosed cluster--class anchors break the permutation ambiguity?
2. Does the transported DP prototype condition add information beyond the OT
   pseudo-label or its one-hot encoding?

Target training labels are accessed only to instantiate the explicitly
supervised public-anchor simulation and to compute post-hoc assignment accuracy.
They are never used by the zero-anchor method, the empirical reliability screen, or
the condition-isolation controls.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t

from core.ot_alignment import PartialOTAligner
from evaluation.downstream import DownstreamTrainer
from component_study import DEFAULT_OUT, SEEDS, apply_variant, args_for
from main import build_config
from trainer import FedKAPTTrainer


def row_plan(aligner):
    plan = np.asarray(aligner.T_soft, dtype=np.float64)
    return plan / (plan.sum(axis=1, keepdims=True) + 1e-12)


def constrained_assignment(score, anchors=()):
    """Best bijection and best-versus-second-best gap under fixed anchors."""
    score = np.asarray(score, dtype=np.float64)
    n_rows, n_cols = score.shape
    if n_rows != n_cols:
        raise ValueError("The assignment-gap study requires equal cardinality")
    fixed = {int(r): int(c) for r, c in anchors}
    if len(set(fixed.values())) != len(fixed):
        raise ValueError("Anchors must use distinct source classes")
    free_rows = [r for r in range(n_rows) if r not in fixed]
    fixed_cols = set(fixed.values())
    free_cols = [c for c in range(n_cols) if c not in fixed_cols]

    mapping = np.full(n_rows, -1, dtype=np.int64)
    for r, c in fixed.items():
        mapping[r] = c
    if free_rows:
        rr, cc = linear_sum_assignment(-score[np.ix_(free_rows, free_cols)])
        for i, j in zip(rr, cc):
            mapping[free_rows[i]] = free_cols[j]
    best = float(score[np.arange(n_rows), mapping].sum())

    alternatives = []
    for row in free_rows:
        forbidden_col = int(mapping[row])
        sub = score[np.ix_(free_rows, free_cols)].copy()
        sub[free_rows.index(row), free_cols.index(forbidden_col)] = -1e12
        rr, cc = linear_sum_assignment(-sub)
        alt_mapping = mapping.copy()
        for i, j in zip(rr, cc):
            alt_mapping[free_rows[i]] = free_cols[j]
        if alt_mapping[row] != forbidden_col:
            alternatives.append(
                float(score[np.arange(n_rows), alt_mapping].sum()))
    second = max(alternatives) if alternatives else float("-inf")
    return mapping, best - second if np.isfinite(second) else float("inf")


def oracle_anchor_order(assignments, labels, n_classes):
    """Deterministic anchor order derived from a one-to-one majority mapping.

    This function is used only to simulate disclosed public correspondences.
    Classes are revealed in ascending class-index order to avoid selecting
    anchors according to downstream performance.
    """
    counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    for cluster in range(n_classes):
        counts[cluster] = np.bincount(
            labels[assignments == cluster], minlength=n_classes)
    rows, cols = linear_sum_assignment(-counts)
    pairs = sorted(zip(rows.tolist(), cols.tolist()), key=lambda rc: rc[1])
    return pairs


def sample_assignment_accuracy(mapping, assignments, labels):
    return float(np.mean(np.asarray(mapping)[assignments] == labels))


def ci95(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(t.ppf(.975, len(values) - 1) *
                 values.std(ddof=1) / math.sqrt(len(values)))


def reliability_records(trainer, perturbation_counts, anchors=()):
    """Post-release local-stability records under added public noise.

    The target perturbs the already released prototypes at the declared DP
    noise scale. This is post-processing and consumes no additional privacy
    budget. The resulting deviation is local to the already private release;
    it is not an estimate of private-versus-nonprivate plan error. Prefixes of
    one shared perturbation stream provide the B-sensitivity check.
    """
    base_plan = row_plan(trainer.aligner)
    base_mapping, gap = constrained_assignment(base_plan, anchors)
    rng = np.random.default_rng(trainer.cfg.seed + 33091)
    deviations, same = [], []
    released = trainer.d_bank.transmit()
    sigma = np.asarray(trainer.d_bank.sigma, dtype=np.float64)
    for _ in range(max(perturbation_counts)):
        perturbed = released + rng.normal(0.0, sigma, released.shape)
        aligner = PartialOTAligner(trainer.cfg).fit(
            trainer.t_bank.raw_prototypes, perturbed.astype(np.float32))
        plan = row_plan(aligner)
        mapping, _ = constrained_assignment(plan, anchors)
        deviations.append(float(np.max(np.abs(plan - base_plan))))
        same.append(bool(np.array_equal(mapping, base_mapping)))
    radius = float(gap / (2.0 * len(base_mapping)))
    records = []
    for count in perturbation_counts:
        eta95 = float(np.quantile(deviations[:count], .95))
        records.append({
            "perturbations": int(count),
            "gap": float(gap),
            "stability_radius": radius,
            "eta95": eta95,
            "accepted": bool(eta95 < radius),
            "exact_assignment_stability": float(np.mean(same[:count])),
            "assignment_accuracy": sample_assignment_accuracy(
                base_mapping, trainer.t_bank.assignments,
                trainer.data.t_train_y),
        })
    return records


def condition_controls(trainer):
    proto_d = trainer.d_bank.transmit()
    train_assign = trainer.t_bank.assignments
    test_assign = trainer._assign_test_clusters(trainer.data.t_test_x)
    pseudo_train = trainer.aligner.get_pseudo_labels(train_assign)
    pseudo_test = trainer.aligner.get_pseudo_labels(test_assign)
    hard = getattr(trainer.cfg.ot, "hard_condition", False)
    cond_train = trainer.aligner.compute_transport_conditions(
        proto_d, train_assign, hard=hard, uniform=False,
        sample_features=trainer.data.t_train_x)
    cond_test = trainer.aligner.compute_transport_conditions(
        proto_d, test_assign, hard=hard, uniform=False,
        sample_features=trainer.data.t_test_x)
    eye = np.eye(trainer.cfg.data.n_classes, dtype=np.float32)
    features = {
        "Transport-Label-Only": (
            trainer.data.t_train_x, trainer.data.t_test_x),
        "One-Hot-Condition": (
            np.concatenate([trainer.data.t_train_x, eye[pseudo_train]], axis=1),
            np.concatenate([trainer.data.t_test_x, eye[pseudo_test]], axis=1)),
        "DP-Prototype-Condition": (
            np.concatenate([trainer.data.t_train_x, cond_train], axis=1),
            np.concatenate([trainer.data.t_test_x, cond_test], axis=1)),
    }
    results = {}
    for name, (x_train, x_test) in features.items():
        model = DownstreamTrainer(trainer.cfg, name)
        model.train(x_train, pseudo_train, trainer.logger)
        results[name] = model.evaluate(x_test, trainer.data.t_test_y)
    return results


def run(out_dir: Path, seeds, perturbation_counts):
    path = out_dir / "identifiability_study.json"
    previous = (json.loads(path.read_text(encoding="utf-8"))
                if path.exists() else {})
    payload = {
        "protocol": {
            "version": 3,
            "seeds": list(seeds),
            "anchor_counts": [0, 1],
            "perturbation_counts": list(perturbation_counts),
            "primary_perturbations": int(max(perturbation_counts)),
            "reliability_rule": (
                "post-release privacy-scale local stability: eta95 < Gamma/(2K)"),
            "cardinality_scope": "equal-cardinality Hungarian assignment only",
            "anchor_order": "ascending class index under one-to-one majority mapping",
        },
        "reliability": {
            "cwru": previous.get("reliability", {}).get("cwru", []),
            "wdbc": [],
        },
        "anchor_reliability": [],
        "condition_controls": {
            "cwru": previous.get("condition_controls", {}).get("cwru", []),
            "wdbc": [],
        },
        "anchors": [],
    }
    for dataset in ("wdbc",):
        for seed in seeds:
            print(f"THIRD ROUND {dataset} seed={seed}", flush=True)
            ns = args_for(seed, dataset, out_dir)
            ns.exp_name = f"identifiability_{dataset}_s{seed}"
            cfg = build_config(ns, dataset)
            apply_variant(cfg, "FedKAPT")
            trainer = FedKAPTTrainer(cfg)
            trainer.phase1_prototypes()
            trainer.phase2_ot_alignment()

            rel_records = reliability_records(
                trainer, perturbation_counts)
            for rel in rel_records:
                rel["seed"] = int(seed)
            payload["reliability"][dataset].extend(rel_records)

            controls = condition_controls(trainer)
            payload["condition_controls"][dataset].append({
                "seed": int(seed), "metrics": controls})

            if dataset == "wdbc":
                plan = row_plan(trainer.aligner)
                anchor_order = oracle_anchor_order(
                    trainer.t_bank.assignments, trainer.data.t_train_y,
                    trainer.cfg.data.n_classes)
                for count in (0, 1):
                    anchors = anchor_order[:count]
                    mapping, gap = constrained_assignment(plan, anchors)
                    anchor_rel = reliability_records(
                        trainer, perturbation_counts, anchors)
                    for rel in anchor_rel:
                        rel.update({
                            "seed": int(seed),
                            "n_anchors": int(count),
                            "anchors": [[int(r), int(c)]
                                        for r, c in anchors],
                        })
                    payload["anchor_reliability"].extend(anchor_rel)
                    trainer.aligner.cluster_to_class = mapping
                    gen_tr, var_tr, cond_tr, pseudo, gen_te = trainer.phase3_generate()
                    weights = trainer.phase4_filter(
                        gen_tr, var_tr, cond_tr, pseudo)
                    trainer.phase5_train_classifiers(
                        gen_tr, gen_te, pseudo, weights)
                    metrics = trainer.phase5_evaluate()["FedKAPT"]
                    payload["anchors"].append({
                        "seed": int(seed), "n_anchors": int(count),
                        "anchors": [[int(r), int(c)] for r, c in anchors],
                        "assignment_accuracy": sample_assignment_accuracy(
                            mapping, trainer.t_bank.assignments,
                            trainer.data.t_train_y),
                        "assignment_gap": float(gap),
                        "metrics": metrics,
                    })
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Lightweight summaries make manuscript transfer less error-prone.
    payload["summary"] = {
        "condition_controls": {}, "anchors": {},
        "anchor_reliability": {}, "reliability_sensitivity": {}}
    for dataset in ("cwru", "wdbc"):
        payload["summary"]["condition_controls"][dataset] = {}
        for name in ("Transport-Label-Only", "One-Hot-Condition",
                     "DP-Prototype-Condition"):
            vals = [r["metrics"][name]["accuracy"]
                    for r in payload["condition_controls"][dataset]]
            payload["summary"]["condition_controls"][dataset][name] = {
                "mean_accuracy": float(np.mean(vals)),
                "ci95": ci95(vals), "values": vals}
    for count in (0, 1):
        rows = [r for r in payload["anchors"] if r["n_anchors"] == count]
        payload["summary"]["anchors"][str(count)] = {
            "assignment_accuracy": float(np.mean(
                [r["assignment_accuracy"] for r in rows])),
            "downstream_accuracy": float(np.mean(
                [r["metrics"]["accuracy"] for r in rows])),
            "assignment_gap": float(np.mean([r["assignment_gap"] for r in rows])),
        }
        payload["summary"]["anchor_reliability"][str(count)] = {}
        for B in perturbation_counts:
            rel_rows = [r for r in payload["anchor_reliability"]
                        if r["n_anchors"] == count
                        and r["perturbations"] == B]
            payload["summary"]["anchor_reliability"][str(count)][str(B)] = {
                "mean_stability_radius": float(np.mean(
                    [r["stability_radius"] for r in rel_rows])),
                "mean_eta95": float(np.mean([r["eta95"] for r in rel_rows])),
                "accept_rate": float(np.mean([r["accepted"] for r in rel_rows])),
                "mean_assignment_accuracy": float(np.mean(
                    [r["assignment_accuracy"] for r in rel_rows])),
            }
    for dataset in ("cwru", "wdbc"):
        payload["summary"]["reliability_sensitivity"][dataset] = {}
        for B in perturbation_counts:
            rows = [r for r in payload["reliability"][dataset]
                    if r["perturbations"] == B]
            payload["summary"]["reliability_sensitivity"][dataset][str(B)] = {
                "accept_rate": float(np.mean([r["accepted"] for r in rows])),
                "mean_eta95": float(np.mean([r["eta95"] for r in rows])),
                "mean_exact_stability": float(np.mean(
                    [r["exact_assignment_stability"] for r in rows])),
            }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--perturbations", nargs="+", type=int,
                        default=[50, 100, 200])
    args = parser.parse_args()
    result = run(Path(args.out_dir), args.seeds, args.perturbations)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
