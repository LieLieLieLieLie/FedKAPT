"""KBS pre-submission diagnostics driven by saved or newly computed data.

The script never fabricates observations. It stores every completed run before
continuing so an interrupted study can be resumed without discarding evidence.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import t, ttest_rel
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ot_alignment import PartialOTAligner
from core.prototype import unlabeled_prototypes
from feddata import get_data_module
from main import build_config
from trainer import FedKAPTTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "models"
SEEDS = [42, 123, 456, 789, 2026]
METRICS = ("accuracy", "macro_f1", "macro_auc")


def args_for(seed, dataset, out_dir, source=None, target=None):
    return SimpleNamespace(
        source=source or "product", target=target or "real_world",
        cwru_source=0 if source is None else int(source),
        cwru_target=2 if target is None else int(target),
        data_dir="./data/office_home", cwru_data_dir="./data/cwru",
        wdbc_data_dir="./data/wdbc",
        epsilon=10.0, delta=1e-5, max_norm=1.0,
        ot_mass=None, ot_reg=0.05, latent_dim=128,
        cvae_epochs=50, downstream_epochs=100,
        beta=1.0, ot_lambda=0.1, seed=int(seed),
        exp_name=None, log_dir=str(out_dir / "logs"),
        save_dir=str(out_dir / "checkpoints"),
        results_dir=str(out_dir / "results"))


VARIANTS = {
    "FedKAPT-Gen": {},
    "FedKAPT-Core (condition only)": {"condition_only": True},
    "w/o latent cycle": {"latent_cycle": 0.0},
    "w/o generator Sinkhorn": {"ot_lambda": 0.0},
    "w/o soft conditioning": {"hard_condition": True},
    "w/o filtering": {"no_filter": True},
    "w/o alignment head": {"no_align": True},
    "w/o late fusion": {"no_late": True},
}


def apply_variant(cfg, variant):
    spec = ({"condition_only": True}
            if variant in ("FedKAPT", "FedKAPT") else VARIANTS[variant])
    if spec.get("condition_only"):
        cfg.cvae.disable_generation = True
        # A deterministic prototype condition has no predictive uncertainty to
        # filter; retaining percentile filtering would assign arbitrary weights
        # to numerical ties and would not be a meaningful core algorithm.
        cfg.filter.min_keep_ratio = 1.0
        cfg.filter.recon_uncertainty_pct = 100.0
        cfg.filter.sem_uncertainty_pct = 100.0
    if "latent_cycle" in spec:
        cfg.cvae.latent_cycle_weight = spec["latent_cycle"]
    if "ot_lambda" in spec:
        cfg.cvae.ot_lambda = spec["ot_lambda"]
    if spec.get("hard_condition"):
        cfg.ot.hard_condition = True
        cfg.ot.use_proto_condition = False
        cfg.ot.nn_condition_weight = 0.0
    if spec.get("no_filter"):
        cfg.filter.min_keep_ratio = 1.0
        cfg.filter.recon_uncertainty_pct = 100.0
        cfg.filter.sem_uncertainty_pct = 100.0
    if spec.get("no_align"):
        cfg.downstream.use_align_fusion = False
    if spec.get("no_late"):
        cfg.downstream.fusion_alpha = 1.0
        cfg.downstream.auto_fusion_alpha = False
        cfg.downstream.use_align_fusion = False


def summarize_variant_runs(runs):
    summary = {}
    for dataset, dataset_runs in runs.items():
        summary[dataset] = {}
        for variant in VARIANTS:
            blocks = [r[variant] for r in dataset_runs if variant in r]
            entry = {}
            for metric in METRICS:
                vals = np.asarray([b["metrics"][metric] for b in blocks], float)
                n = len(vals)
                half = (float(t.ppf(0.975, n - 1)) * vals.std(ddof=1) / math.sqrt(n)
                        if n > 1 else 0.0)
                entry[metric] = {
                    "n": n, "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if n > 1 else 0.0,
                    "ci95": [float(vals.mean() - half), float(vals.mean() + half)],
                    "values": vals.tolist()}
            diag_keys = sorted({
                key for b in blocks for key in (b.get("diagnostics") or {})})
            entry["diagnostics"] = {}
            for key in diag_keys:
                vals = np.asarray([
                    b.get("diagnostics", {}).get(key, np.nan) for b in blocks], float)
                vals = vals[np.isfinite(vals)]
                entry["diagnostics"][key] = {
                    "n": int(len(vals)),
                    "mean": float(vals.mean()) if len(vals) else None,
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
            summary[dataset][variant] = entry
        full = summary[dataset]["FedKAPT-Gen"]
        for variant, entry in summary[dataset].items():
            if variant == "FedKAPT-Gen":
                continue
            entry["paired_vs_full"] = {}
            for metric in METRICS:
                x = np.asarray(full[metric]["values"])
                y = np.asarray(entry[metric]["values"])
                d = y - x
                n = len(d)
                half = (float(t.ppf(0.975, n - 1)) * d.std(ddof=1) / math.sqrt(n)
                        if n > 1 else 0.0)
                entry["paired_vs_full"][metric] = {
                    "mean_difference": float(d.mean()),
                    "ci95": [float(d.mean() - half), float(d.mean() + half)],
                    "p": float(ttest_rel(y, x).pvalue) if n > 1 else None,
                    "win_rate": float(np.mean(y > x))}
    return summary


def run_multiseed_ablation(out_dir, seeds):
    path = out_dir / "component_runs.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if payload.get("protocol", {}).get("version") not in (2, 3):
        payload = {
        "protocol": {"version": 3, "seeds": seeds, "cvae_epochs": 50,
                     "downstream_epochs": 100},
        "runs": {"cwru": [], "wdbc": []}}
    payload["protocol"]["version"] = 3
    payload["protocol"]["datasets"] = ["cwru", "wdbc"]
    payload.setdefault("runs", {}).setdefault("cwru", [])
    payload["runs"].setdefault("wdbc", [])
    payload["runs"].pop("office_home", None)
    for dataset in ("cwru", "wdbc"):
        completed = {int(r["seed"]): r for r in payload["runs"][dataset]}
        for seed in seeds:
            run = completed.get(seed, {"seed": seed})
            for variant in VARIANTS:
                if variant in run:
                    continue
                print(f"ABLATION {dataset} seed={seed} variant={variant}", flush=True)
                ns = args_for(seed, dataset, out_dir)
                ns.exp_name = (f"component_{dataset}_s{seed}_" +
                               variant.lower().replace(" ", "_").replace("/", "_"))
                cfg = build_config(ns, dataset)
                apply_variant(cfg, variant)
                trainer = FedKAPTTrainer(cfg)
                result = trainer.train_test()["FedKAPT"]
                run[variant] = {
                    "metrics": result,
                    "diagnostics": trainer.generation_diagnostics,
                    "selected_mass": float(trainer.aligner.mass)}
                completed[seed] = run
                payload["runs"][dataset] = [completed[s] for s in sorted(completed)]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary"] = summarize_variant_runs(payload["runs"])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def map_accuracy(aligner, assignments, labels):
    pred = np.asarray(aligner.cluster_to_class)[assignments]
    return float(np.mean(pred == labels))


def plan_confidence(aligner):
    plan = np.asarray(aligner.T_soft, float)
    plan /= plan.sum(axis=1, keepdims=True) + 1e-12
    return float(plan.max(axis=1).mean())


def run_sensitivity(out_dir, seeds):
    path = out_dir / "structural_sensitivity.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        records = {
            key: [row for row in previous.get(key, [])
                  if row.get("dataset") == "cwru"]
            for key in ("signature", "mass", "cardinality",
                        "clustering", "corruption")
        }
    else:
        records = {"signature": [], "mass": [], "cardinality": [],
                   "clustering": [], "corruption": []}
    for dataset in ("wdbc",):
        for seed in seeds:
            ns = args_for(seed, dataset, out_dir)
            cfg = build_config(ns, dataset)
            trainer = FedKAPTTrainer(cfg)
            trainer.phase1_prototypes()
            data = trainer.data
            proto_t = trainer.t_bank.raw_prototypes
            assignments = trainer.t_bank.assignments.astype(int)
            proto_d = trainer.d_bank.transmit()

            for length in (3, 4, 8, 16, 32, 64):
                cfg.ot.signature_dim = length
                aligner = PartialOTAligner(cfg).fit(proto_t, proto_d)
                records["signature"].append({
                    "dataset": dataset, "seed": seed, "L": length,
                    "accuracy": map_accuracy(
                        aligner, assignments, data.t_train_y),
                    "selected_mass": float(aligner.mass)})

            cfg.ot.signature_dim = 32
            mass_rows = []
            for mass in cfg.ot.mass_search_grid:
                cfg.ot.partial_mass = mass
                aligner = PartialOTAligner(cfg).fit(proto_t, proto_d)
                row = {"dataset": dataset, "seed": seed, "mass": mass,
                       "confidence": plan_confidence(aligner),
                       "accuracy": map_accuracy(
                           aligner, assignments, data.t_train_y)}
                records["mass"].append(row)
                mass_rows.append(row)
            cfg.ot.partial_mass = None

            counts = ((3, 4, 5, 6) if dataset == "cwru" else (1, 2, 3))
            for count in counts:
                centers, assign, _ = unlabeled_prototypes(
                    data.t_train_x, count, seed=seed, algorithm="kmeans")
                cfg.prototype.n_clusters = count
                aligner = PartialOTAligner(cfg).fit(centers, proto_d)
                records["cardinality"].append({
                    "dataset": dataset, "seed": seed,
                    "clusters": count,
                    "ratio": count / cfg.data.n_classes,
                    "accuracy": map_accuracy(aligner, assign, data.t_train_y)})

            cfg.prototype.n_clusters = cfg.data.n_classes
            for algorithm in ("kmeans", "agglomerative", "gmm"):
                centers, assign, _ = unlabeled_prototypes(
                    data.t_train_x, cfg.data.n_classes,
                    seed=seed, algorithm=algorithm)
                aligner = PartialOTAligner(cfg).fit(centers, proto_d)
                records["clustering"].append({
                    "dataset": dataset, "seed": seed,
                    "algorithm": algorithm,
                    "accuracy": map_accuracy(aligner, assign, data.t_train_y)})

            x_norm = normalize(data.t_train_x)
            rng = np.random.default_rng(seed + 9102)
            for rate in (0.0, 0.1, 0.2, 0.3, 0.4):
                corrupt = assignments.copy()
                n_change = int(round(rate * len(corrupt)))
                if n_change:
                    idx = rng.choice(len(corrupt), n_change, replace=False)
                    offset = rng.integers(1, cfg.data.n_classes, size=n_change)
                    corrupt[idx] = (corrupt[idx] + offset) % cfg.data.n_classes
                centers = np.stack([
                    x_norm[corrupt == k].mean(axis=0)
                    if np.any(corrupt == k) else proto_t[k]
                    for k in range(cfg.data.n_classes)])
                aligner = PartialOTAligner(cfg).fit(centers, proto_d)
                records["corruption"].append({
                    "dataset": dataset, "seed": seed, "rate": rate,
                    "accuracy": map_accuracy(
                        aligner, corrupt, data.t_train_y)})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_sensitivity", action="store_true")
    parser.add_argument(
        "--reset_wdbc", action="store_true",
        help="Discard cached WDBC component runs before a protocol correction.")
    ns = parser.parse_args()
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ns.reset_wdbc:
        component_path = out_dir / "component_runs.json"
        if component_path.exists():
            cached = json.loads(component_path.read_text(encoding="utf-8"))
            cached.setdefault("runs", {})["wdbc"] = []
            cached.pop("summary", None)
            component_path.write_text(
                json.dumps(cached, indent=2), encoding="utf-8")
    if not ns.skip_ablation:
        run_multiseed_ablation(out_dir, ns.seeds)
    if not ns.skip_sensitivity:
        run_sensitivity(out_dir, ns.seeds)
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
