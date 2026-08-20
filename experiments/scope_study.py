"""Resumable CWRU-direction and WDBC-partition scope validation."""

import argparse
import json
from pathlib import Path

from component_study import DEFAULT_OUT, apply_variant, args_for
from main import build_config
from trainer import FedKAPTTrainer


CWRU_PAIRS = [(s, t) for s in (0, 1, 2, 3) for t in (0, 1, 2, 3) if s != t]
WDBC_PARTITIONS = [20260819, 20260820, 20260821, 20260822]


def run(out_dir: Path, seeds):
    path = out_dir / "scope_runs.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = {
        "protocol": {
            "method": "FedKAPT",
            "seeds": seeds,
            "cwru_pairs": [f"{s}->{t}" for s, t in CWRU_PAIRS],
            "wdbc_partition_seeds": WDBC_PARTITIONS,
        },
        "runs": {
            "cwru": previous.get("runs", {}).get("cwru", {}),
            "wdbc": {},
        },
    }
    study_blocks = (
        ("cwru", CWRU_PAIRS),
        ("wdbc", [(partition, "mean-to-worst")
                  for partition in WDBC_PARTITIONS]),
    )
    for dataset, pairs in study_blocks:
        for src, tgt in pairs:
            tag = (f"{src}->{tgt}" if dataset == "cwru"
                   else f"partition-{src}")
            payload["runs"][dataset].setdefault(tag, {})
            for seed in seeds:
                skey = str(seed)
                if skey in payload["runs"][dataset][tag]:
                    continue
                ns = args_for(seed, dataset, out_dir)
                ns.exp_name = f"scope_{dataset}_{src}_to_{tgt}_s{seed}"
                cfg = (build_config(ns, dataset, src, tgt)
                       if dataset == "cwru" else build_config(ns, dataset))
                if dataset == "wdbc":
                    cfg.data.wdbc_partition_seed = int(src)
                apply_variant(cfg, "FedKAPT")
                print(f"SCOPE {dataset} {tag} seed={seed}", flush=True)
                trainer = FedKAPTTrainer(cfg)
                result = trainer.train_test()
                payload["runs"][dataset][tag][skey] = {
                    "Baseline": result["Baseline"],
                    "FedKAPT": result["FedKAPT"],
                    "selected_mass": float(trainer.aligner.mass),
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ns = parser.parse_args()
    out_dir = Path(ns.out_dir)
    run(out_dir, ns.seeds)
    print((out_dir / "scope_runs.json").resolve())


if __name__ == "__main__":
    main()
