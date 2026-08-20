"""Print concise statistics used in the FedKAPT manuscript."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results" / "models"

abl = json.loads((ART / "component_runs.json").read_text())
for ds in ("cwru", "wdbc"):
    print("\n", ds)
    deterministic_name = ("FedKAPT" if "FedKAPT" in abl["summary"][ds]
                          else "FedKAPT-Core (condition only)")
    for name in ("FedKAPT-Gen", deterministic_name,
                 "w/o latent cycle", "w/o generator Sinkhorn",
                 "w/o soft conditioning", "w/o filtering",
                 "w/o alignment head", "w/o late fusion"):
        row = abl["summary"][ds][name]
        vals = [100 * row[m]["mean"] for m in
                ("accuracy", "macro_f1", "macro_auc")]
        print(f"{name:34s} {vals[0]:.2f} {vals[1]:.2f} {vals[2]:.2f}")
    diag = abl["summary"][ds]["FedKAPT-Gen"]["diagnostics"]
    print("sample specificity", diag["sample_specificity_ratio"])
    print("condition RMSE", diag["condition_rmse"])
    print("latent cycle", diag["latent_cycle_mse"])

scope = json.loads((ART / "scope_summary.json").read_text())
for ds in scope:
    print("\nSCOPE", ds["dataset"], ds["pairs"], ds["runs"],
          ds["mean_accuracy"], ds["mean_gain"])
    for row in ds["pair_results"]:
        print(row)
