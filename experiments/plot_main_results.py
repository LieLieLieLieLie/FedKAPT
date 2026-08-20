"""Build the audit figures and LaTeX tables used by the KBS manuscript.

All panels are derived from saved experiment JSON.  The script intentionally
does not invent, smooth, or interpolate observations.  Main comparisons use
ten paired seeds; privacy and component ablations are labelled as single-seed
diagnostics in the manuscript.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "models" / "audit_results.json"
PAPER = ROOT.parent / "paper" / "KBS"
FIGDIR = ROOT / "results" / "figures"
PAPER_FIGDIR = PAPER / "figures" / "experiment"

METHODS = [
    "NoTransfer", "CORAL-FTL", "LS-ProtoAlign", "Entropy-ST",
    "DP-ProtoAug", "Rel-FullOT", "Rel-UOT", "GW-FTL", "FedKAPT (Ours)",
]
OTHER_COLORS = ["#FFAA53", "#50CC55", "#3399FF", "#6666FF",
                "#9933FF", "#00DDDD", "#4D4D4D"]
COLORS = {method: OTHER_COLORS[i % len(OTHER_COLORS)]
          for i, method in enumerate(METHODS[:-1])}
COLORS["FedKAPT (Ours)"] = "#FF6666"
POSITIVE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "positive_white_red", ["#FFFFFF", "#FF4F4F"], N=256)
METRICS = ["accuracy", "macro_f1", "macro_auc"]
METRIC_LABELS = ["Accuracy", "Macro-F1", "Macro-AUC"]
DATA_LABELS = {"cwru": "CWRU control", "wdbc": "WDBC clinical transfer"}


def load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def configure():
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9.0, "axes.labelsize": 9.0, "axes.titlesize": 9.0,
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel(ax, letter):
    ax.text(-0.10, 1.06, f"({letter})", transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", clip_on=False)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.5, zorder=0)


def ci_half(summary, method, metric):
    item = summary[method][metric]
    return (item["ci95"][1] - item["ci95"][0]) / 2.0


def plot_method_ci(ax, summary, metric, letter, dataset):
    vals = [summary[m][metric]["mean"] for m in METHODS]
    errs = [ci_half(summary, m, metric) for m in METHODS]
    y = np.arange(len(METHODS))
    for yi, method, val, err in zip(y, METHODS, vals, errs):
        ax.errorbar(val, yi, xerr=err, fmt="o", ms=4.0,
                    color=COLORS[method], ecolor=COLORS[method],
                    elinewidth=1.0, capsize=2.2, zorder=3)
    ax.set_yticks(y)
    short = ["NoT", "CORAL", "LS", "EST", "DP", "FOT", "UOT", "GW",
             "FedKAPT"]
    ax.set_yticklabels(short)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy [fraction]")
    ax.set_title(("CWRU" if dataset == "cwru" else "WDBC") +
                 " mean and 95% CI")
    lo = max(0.0, min(v - e for v, e in zip(vals, errs)) - 0.04)
    hi = min(1.0, max(v + e for v, e in zip(vals, errs)) + 0.04)
    ax.set_xlim(lo, hi)
    panel(ax, letter)


def paired_delta(ax, runs, dataset, letter):
    ours = np.asarray([r["FedKAPT (Ours)"]["accuracy"] for r in runs])
    if dataset == "wdbc":
        competitors = ["NoTransfer", "CORAL-FTL", "Entropy-ST",
                       "LS-ProtoAlign"]
        tick_labels = ["NoT", "CORAL", "EST", "LS"]
    else:
        competitors = ["NoTransfer", "LS-ProtoAlign", "DP-ProtoAug",
                       "GW-FTL"]
        tick_labels = ["NoT", "LS", "DP", "GW"]
    rng = np.random.default_rng(1729)
    deltas = [ours - np.asarray([r[m]["accuracy"] for r in runs]) for m in competitors]
    bp = ax.boxplot(deltas, positions=np.arange(len(competitors)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops={"color": "#222222", "linewidth": 1.0})
    for box, method in zip(bp["boxes"], competitors):
        box.set_facecolor(COLORS[method]); box.set_alpha(0.45)
    for i, ds in enumerate(deltas):
        jitter = rng.uniform(-0.13, 0.13, size=len(ds))
        ax.scatter(i + jitter, ds, s=11, color="#222222", alpha=0.65, zorder=3)
    ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xticks(np.arange(len(competitors)))
    ax.set_xticklabels(tick_labels, rotation=18)
    ax.set_ylabel(r"Paired $\Delta$ accuracy [fraction]")
    ax.set_title(("CWRU" if dataset == "cwru" else "WDBC") +
                 " paired difference")
    panel(ax, letter)


def win_heatmap(ax, datasets, letter):
    competitors = METHODS[:-1]
    cols = []
    labels = []
    for ds in ["cwru", "wdbc"]:
        for metric, metric_label in zip(METRICS, METRIC_LABELS):
            cols.append([datasets[ds]["summary"][m][metric]["fedkapt_win_rate"]
                         for m in competitors])
            labels.append(("C" if ds == "cwru" else "W") + "-" + metric_label.replace("Macro-", ""))
    arr = np.asarray(cols).T
    im = ax.imshow(arr, cmap=POSITIVE_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center",
                    fontsize=7.2, color="black" if arr[i, j] < 0.62 else "white")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_yticks(np.arange(len(competitors)))
    ax.set_yticklabels(["NoT", "CORAL", "LS", "EST", "DP", "FOT", "UOT", "GW"])
    ax.set_title("FedKAPT win rate")
    ax.grid(False)
    panel(ax, letter)
    return im


def rank_panel(ax, datasets, letter):
    x = np.arange(len(METHODS))
    width = 0.35
    for off, ds, color in [(-width / 2, "cwru", "#FFAA53"),
                           (width / 2, "wdbc", "#3399FF")]:
        runs = datasets[ds]["runs"]
        ranks = []
        for run in runs:
            mean_score = np.asarray([np.mean([run[m][k] for k in METRICS]) for m in METHODS])
            order = np.argsort(-mean_score)
            r = np.empty_like(order, dtype=float); r[order] = np.arange(1, len(METHODS) + 1)
            ranks.append(r)
        ranks = np.asarray(ranks)
        means = ranks.mean(axis=0)
        ci = 1.96 * ranks.std(axis=0, ddof=1) / math.sqrt(len(ranks))
        ax.bar(x + off, means, width, yerr=ci, color=color, alpha=0.78,
               capsize=2, label=("CWRU" if ds == "cwru" else "WDBC"), zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(["NoT", "CORAL", "LS", "EST", "DP", "FOT", "UOT", "GW", "FedKAPT"], rotation=55, ha="right")
    ax.set_ylabel("Average rank [1=best]")
    ax.set_title("Mean-score rank")
    ax.set_ylim(0, 12)
    ax.set_yticks(np.arange(0, 13, 2))
    ax.legend(frameon=False, ncol=1, loc="upper left",
              bbox_to_anchor=(0.01, 0.99), borderaxespad=0,
              handlelength=1.5, labelspacing=0.25)
    panel(ax, letter)


def save_main(audit):
    datasets = audit["datasets"]
    # The 2x3 main-evidence composite spans both columns so that every panel
    # retains its original canvas size at the journal's final print scale.
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 6.0), constrained_layout=True)
    plot_method_ci(axes[0, 0], datasets["cwru"]["summary"], "accuracy", "a", "cwru")
    plot_method_ci(axes[0, 1], datasets["wdbc"]["summary"], "accuracy", "b", "wdbc")
    paired_delta(axes[0, 2], datasets["cwru"]["runs"], "cwru", "c")
    paired_delta(axes[1, 0], datasets["wdbc"]["runs"], "wdbc", "d")
    win_heatmap(axes[1, 1], datasets, "e")
    rank_panel(axes[1, 2], datasets, "f")
    for ax in axes.flat:
        plt.setp(ax.get_yticklabels(), rotation=60, ha="right",
                 rotation_mode="anchor")
    fig.savefig(FIGDIR / "main_evidence_composite.pdf", bbox_inches="tight")
    plt.close(fig)
    if PAPER.is_dir():
        PAPER_FIGDIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIGDIR / "main_evidence_composite.pdf",
                     PAPER_FIGDIR / "main_evidence_composite.pdf")


def extract_ablation(path):
    raw = load(path)
    out = {}
    for name, block in raw.items():
        m = block.get("FedKAPT", {})
        if m:
            out[name] = [m.get(k, np.nan) for k in METRICS]
    return out


def privacy_panel(ax, raw, letter, title):
    labels = [k for k in ["0.1", "0.5", "1.0", "2.0", "5.0", "10.0", "inf"] if k in raw]
    x = np.arange(len(labels))
    for metric, lab, color, marker in zip(
            METRICS, METRIC_LABELS, ["#FFAA53", "#50CC55", "#3399FF"], ["o", "s", "^"]):
        vals = [raw[k][metric] for k in labels]
        ax.plot(x, vals, marker=marker, ms=3.5, lw=1.2, color=color, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["No DP" if k == "inf" else k for k in labels], rotation=25)
    ax.set_xlabel(r"Privacy budget $\epsilon$ [dimensionless]")
    ax.set_ylabel("Score [0--1]")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    panel(ax, letter)


def ablation_panel(ax, abl, letter, title):
    names = list(abl.keys())
    full = np.nanmean(abl["Full FedKAPT"])
    drops = [full - np.nanmean(abl[n]) for n in names]
    colors = ["#FF6666" if n == "Full FedKAPT" else
              OTHER_COLORS[i % len(OTHER_COLORS)] for i, n in enumerate(names)]
    ax.barh(np.arange(len(names)), drops, color=colors, alpha=0.8)
    ax.axvline(0, color="#555555", ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([n.replace("w/o ", r"-$\,$") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Mean-score drop [fraction]")
    ax.set_title(title)
    panel(ax, letter)


def ecdf_panel(ax, audit, letter):
    for ds, color in [("cwru", "#FFAA53"), ("wdbc", "#3399FF")]:
        vals = sorted(audit["datasets"][ds]["summary"]["FedKAPT (Ours)"]["accuracy"]["values"])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=color, lw=1.5, label=DATA_LABELS[ds])
    ax.set_xlabel("FedKAPT accuracy [0--1]")
    ax.set_ylabel("Empirical CDF [fraction]")
    ax.set_title("Run-to-run reliability (10 seeds)")
    ax.legend(frameon=False)
    panel(ax, letter)


def mass_panel(ax, ablations, letter):
    ds_order = ["cwru", "wdbc"]
    x = np.arange(2)
    width = 0.34
    partial = [np.nanmean(ablations[d]["Full FedKAPT"]) for d in ds_order]
    full = [np.nanmean(ablations[d]["Full OT (m=1)"]) for d in ds_order]
    ax.bar(x - width / 2, partial, width, color="#FF6666", label="Selected partial mass")
    ax.bar(x + width / 2, full, width, color="#FFAA53", label="Full OT, m=1")
    ax.set_xticks(x); ax.set_xticklabels(["CWRU", "WDBC"])
    ax.set_ylabel("Mean of three scores [0--1]")
    ax.set_title("Is partial transport necessary?")
    ax.legend(frameon=False)
    panel(ax, letter)


def save_robustness(audit, privacy, ablations):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.8), constrained_layout=True)
    privacy_panel(axes[0, 0], privacy["cwru"], "a", "CWRU privacy response")
    privacy_panel(axes[0, 1], privacy["wdbc"], "b", "WDBC privacy response")
    ablation_panel(axes[0, 2], ablations["cwru"], "c", "CWRU one-factor ablation")
    ablation_panel(axes[1, 0], ablations["wdbc"], "d", "WDBC one-factor ablation")
    ecdf_panel(axes[1, 1], audit, "e")
    mass_panel(axes[1, 2], ablations, "f")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.015))
    fig.savefig(FIGDIR / "robustness_composite.pdf", bbox_inches="tight")
    plt.close(fig)


def pm(mean, std):
    return f"{100*mean:.1f} $\\pm$ {100*std:.1f}"


def write_tables(audit, ablations):
    lines = ["% Auto-generated from verified JSON artifacts; do not edit by hand."]
    for ds in ["cwru", "wdbc"]:
        summary = audit["datasets"][ds]["summary"]
        lines += [f"\\begin{{table*}}[t]", "\\centering", "\\small",
                  f"\\caption{{Ten-seed target-test results on {DATA_LABELS[ds]}. Values are mean $\\pm$ sample standard deviation in percentage points.}}",
                  f"\\label{{tab:main_{ds}}}",
                  "\\begin{tabular}{lccc}", "\\toprule",
                  "Method & Accuracy (\\%) & Macro-F1 (\\%) & Macro-AUC (\\%) \\\\", "\\midrule"]
        for method in METHODS:
            vals = [summary[method][k] for k in METRICS]
            cells = [pm(v["mean"], v["std"]) for v in vals]
            label = "\\textbf{FedKAPT}" if method == "FedKAPT (Ours)" else method
            lines.append(label + " & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]

    (PAPER / "generated_results.tex").write_text("\n".join(lines), encoding="utf-8")


def write_summary(audit, ablations):
    out = ["# Recomputed evidence summary", "", "All main results use 10 paired seeds. Privacy and ablation diagnostics use seed 42.", ""]
    for ds in ["cwru", "wdbc"]:
        out.append(f"## {DATA_LABELS[ds]}")
        for method in METHODS:
            s = audit["datasets"][ds]["summary"][method]
            out.append(f"- {method}: " + ", ".join(
                f"{lab}={100*s[m]['mean']:.2f}+-{100*s[m]['std']:.2f}"
                for m, lab in zip(METRICS, METRIC_LABELS)))
        out.append("")
        out.append("Ablation (single seed):")
        for name, vals in ablations[ds].items():
            out.append(f"- {name}: " + ", ".join(f"{x:.4f}" for x in vals))
        out.append("")
    (PAPER / "RECOMPUTED_RESULTS.md").write_text("\n".join(out), encoding="utf-8")


def main():
    configure()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    audit = load(AUDIT)
    save_main(audit)
    print(FIGDIR)


if __name__ == "__main__":
    main()
