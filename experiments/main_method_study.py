"""Ten-seed main evaluation of FedKAPT."""

import json
from pathlib import Path

from component_study import DEFAULT_OUT, apply_variant, args_for
from main import build_config
from trainer import FedKAPTTrainer

SEEDS = [42, 123, 456, 789, 2026, 3141, 2718, 8080, 9001, 10007]


def main():
    path = Path(DEFAULT_OUT) / "main_method_10seed.json"
    data = json.loads(path.read_text()) if path.exists() else {
        "protocol": {"seeds": SEEDS, "method": "FedKAPT"},
        "runs": {"cwru": {}, "wdbc": {}},
    }
    for ds in ("cwru", "wdbc"):
        for seed in SEEDS:
            key = str(seed)
            if key in data["runs"][ds]:
                continue
            ns = args_for(seed, ds, Path(DEFAULT_OUT))
            ns.exp_name = f"main_method_{ds}_s{seed}"
            cfg = build_config(ns, ds)
            apply_variant(cfg, "FedKAPT")
            print(f"MAIN METHOD {ds} seed={seed}", flush=True)
            trainer = FedKAPTTrainer(cfg)
            result = trainer.train_test()
            data["runs"][ds][key] = result["FedKAPT"]
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
