# FedKAPT

Official implementation of **FedKAPT: Privacy-Preserving Knowledge Alignment via Prototype Transport for Feature-Disjoint Federated Transfer**.

FedKAPT supports single-round knowledge transfer between a labelled source party and an unlabelled target party when their feature coordinates are disjoint. The implementation separates differentially private prototype release, relational/anchored partial optimal-transport alignment, target-side acceptance, and downstream prediction.

## Repository layout

```text
core/          privacy, prototype, filtering, and OT-alignment modules
feddata/       CWRU and WDBC data loaders
models/        neural-network modules
evaluation/    downstream training and metric utilities
experiments/   benchmark, ablation, scope, identifiability, and plotting scripts
tests/         privacy-contract and implementation-invariant tests
config.py      experiment configuration dataclasses
main.py        common configuration builder and experiment entry point
trainer.py     FedKAPT training and evaluation pipeline
prepare_data.py  dataset download helper
```

Datasets, trained models, logs, spreadsheets, and generated figures are deliberately excluded from version control. Experiment scripts create the local `results/` hierarchy when needed.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

The repository does not redistribute third-party data.

- **CWRU Bearing Data Center:** download the 12 kHz drive-end files from the [official Case Western Reserve University portal](https://engineering.case.edu/bearingdatacenter/download-data-file). The helper downloads the 16 files used by the manuscript into `data/cwru/`.
- **Wisconsin Diagnostic Breast Cancer (WDBC):** obtain `wdbc.data` from the [UCI Machine Learning Repository dataset page](https://archive.ics.uci.edu/dataset/17/breast) or use the helper to place it in `data/wdbc/`.

```bash
python prepare_data.py --dataset all
```

Expected local layout:

```text
data/
  cwru/
    97.mat
    ...
  wdbc/
    wdbc.data
```

## Reproducing the experiments

Run commands from the repository root. The principal workflows are:

```bash
python experiments/benchmark_study.py
python experiments/main_method_study.py
python experiments/merge_results.py
python experiments/component_study.py
python experiments/scope_study.py
python experiments/identifiability_study.py
python experiments/relational_only_study.py
python experiments/report_statistics.py
python experiments/plot_main_results.py
python experiments/plot_analysis_results.py
```

The plotting scripts read saved JSON records from `results/models/` and export PDF figures to `results/figures/`. Table-oriented exports are written to `results/tables/`. Because the repository contains neither data nor result artifacts, the study scripts must be run before the plotting scripts.

## Privacy and protocol note

For WDBC, the source observes ten mean nucleus descriptors and the target observes ten worst-value descriptors on disjoint patient cohorts. For CWRU, source and target correspond to different load conditions. Alignment quantities are computed from the differentially private source prototype message; the implementation rejects `clean_alignment=True` to prevent an accidental bypass of the one-message privacy contract.

## Testing

```bash
python -m pytest -q
```

## Citation

Please cite the accompanying manuscript if you use this implementation. Formal bibliographic metadata will be added after publication.
