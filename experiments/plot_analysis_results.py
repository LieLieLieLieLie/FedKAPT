"""Create KBS-ready composite figures and inferential tables from saved runs."""

import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, ttest_rel


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results" / "models"
PAPER = ROOT.parent / "paper" / "KBS"
FIG = ROOT / "results" / "figures"
PAPER_FIG = PAPER / "figures" / "experiment"

COLORS = {"cwru": "#FFAA53", "wdbc": "#3399FF"}
FEDKAPT_COLOR = "#FF6666"
OTHER_COLORS = ["#FFAA53", "#50CC55", "#3399FF", "#6666FF",
                "#9933FF", "#00DDDD", "#4D4D4D"]
LABELS = {"cwru": "CWRU", "wdbc": "WDBC"}
METRICS = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
           ("macro_auc", "Macro-AUC")]


def setup(single_column=False):
    base = 7.0 if single_column else 8.0
    line_width = .95 if single_column else 1.45
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": base,
        "axes.labelsize": base, "axes.titlesize": base + .3,
        "legend.fontsize": base, "xtick.labelsize": base - .4,
        "ytick.labelsize": base - .4, "axes.linewidth": 0.8,
        "axes.labelpad": 1.5,
        "lines.linewidth": line_width, "lines.markersize": 2.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def ci(vals):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return float(vals.mean()), 0.0
    return (float(vals.mean()),
            float(t.ppf(.975, len(vals) - 1) * vals.std(ddof=1) /
                  math.sqrt(len(vals))))


def panel(ax, letter):
    ax.text(-0.08, 1.05, f"({letter})", transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left", clip_on=False)


def grouped(rows, key):
    out = {}
    for row in rows:
        out.setdefault((row["dataset"], row[key]), []).append(row["accuracy"])
    return out


def structural_figure():
    setup(single_column=True)
    data = json.loads((ART / "structural_sensitivity.json").read_text())
    # Single-column 2x3 composite: changing the float width must not change
    # the semantic panel order or the original two-row layout.
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 3.30))
    axes = axes.ravel()

    # (a) Signature resolution
    ax = axes[0]
    groups = grouped(data["signature"], "L")
    for ds in COLORS:
        xs = sorted(k[1] for k in groups if k[0] == ds)
        ys, es = zip(*(ci(groups[(ds, x)]) for x in xs))
        ax.errorbar(np.arange(len(xs)), ys, yerr=es, marker="o", capsize=2,
                    color=COLORS[ds], label=LABELS[ds])
    ax.set_xticks(np.arange(len(xs)), [str(x) for x in xs])
    ax.set_xlabel("Resolution $L$ [count]")
    ax.set_ylabel("Assign. acc. [fraction]")
    ax.axvline(xs.index(32), color=".55", ls="--", lw=.8)
    panel(ax, "a")

    # (b) Cardinality mismatch
    ax = axes[1]
    groups = grouped(data["cardinality"], "ratio")
    for ds in COLORS:
        xs = sorted(k[1] for k in groups if k[0] == ds)
        ys, es = zip(*(ci(groups[(ds, x)]) for x in xs))
        ax.errorbar(xs, ys, yerr=es, marker="s", capsize=2,
                    color=COLORS[ds], label=LABELS[ds])
    ax.axvline(1, color=".45", ls="--", lw=.8)
    ax.set_xlabel("$C_t/K$ [ratio]")
    ax.set_ylabel("Assign. acc. [fraction]")
    panel(ax, "b")

    # (c) Clustering algorithms
    ax = axes[2]
    algs = ["kmeans", "agglomerative", "gmm"]
    for j, ds in enumerate(COLORS):
        vals = [[r["accuracy"] for r in data["clustering"]
                 if r["dataset"] == ds and r["algorithm"] == alg]
                for alg in algs]
        pos = np.arange(3) + (-.17 if j == 0 else .17)
        bp = ax.boxplot(vals, positions=pos, widths=.28, patch_artist=True,
                        showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=COLORS[ds], alpha=.35, edgecolor=COLORS[ds])
        for med in bp["medians"]:
            med.set(color=COLORS[ds], lw=1.4)
        for p, arr in zip(pos, vals):
            ax.scatter(np.full(len(arr), p), arr, s=10, color=COLORS[ds],
                       alpha=.65, edgecolors="none")
    ax.set_xticks(range(3), ["K-means", "Agglom.", "GMM"],
                  rotation=20, ha="right")
    ax.set_ylabel("Assign. acc. [fraction]")
    panel(ax, "c")

    # (d) Confidence selected by the unsupervised mass heuristic
    ax = axes[3]
    for ds in COLORS:
        xs = sorted({r["mass"] for r in data["mass"] if r["dataset"] == ds})
        ys, es = [], []
        for x in xs:
            vals = [r["confidence"] for r in data["mass"]
                    if r["dataset"] == ds and r["mass"] == x]
            m, e = ci(vals); ys.append(m); es.append(e)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2,
                    color=COLORS[ds], label=LABELS[ds])
    ax.set_xlabel("Mass $m$ [fraction]")
    ax.set_ylabel("Confidence [fraction]")
    panel(ax, "d")

    # (e) Semantic accuracy over the same mass grid
    ax = axes[4]
    for ds in COLORS:
        xs = sorted({r["mass"] for r in data["mass"] if r["dataset"] == ds})
        ys, es = [], []
        for x in xs:
            vals = [r["accuracy"] for r in data["mass"]
                    if r["dataset"] == ds and r["mass"] == x]
            m, e = ci(vals); ys.append(m); es.append(e)
        ax.errorbar(xs, ys, yerr=es, marker="s", capsize=2,
                    color=COLORS[ds], label=LABELS[ds])
    ax.set_xlabel("Mass $m$ [fraction]")
    ax.set_ylabel("Assign. acc. [fraction]")
    panel(ax, "e")

    # (f) Controlled cluster corruption
    ax = axes[5]
    for ds in COLORS:
        xs = sorted({r["rate"] for r in data["corruption"]
                     if r["dataset"] == ds})
        ys, es = [], []
        for x in xs:
            vals = [r["accuracy"] for r in data["corruption"]
                    if r["dataset"] == ds and r["rate"] == x]
            m, e = ci(vals); ys.append(m); es.append(e)
        ax.errorbar(100 * np.asarray(xs), ys, yerr=es, marker="^", capsize=2,
                    color=COLORS[ds], label=LABELS[ds])
    ax.set_xlabel("Corruption [%]")
    ax.set_ylabel("Assign. acc. [fraction]")
    panel(ax, "f")

    for ax in axes:
        ax.set_box_aspect(1)
        ax.grid(True, color="#d9d9d9", lw=.55, alpha=.75)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [mpl.lines.Line2D([], [], color=COLORS[d], marker="o",
                               label=LABELS[d]) for d in COLORS]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(.5, .965))
    fig.subplots_adjust(left=.12, right=.99, bottom=.10, top=.87,
                        wspace=.58, hspace=.22)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "structural_sensitivity_composite.pdf",
                bbox_inches="tight")
    plt.close(fig)


def holm(pvals):
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals), float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (len(pvals) - rank) * pvals[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_table():
    audit = json.loads((ART / "audit_results.json").read_text())
    rows = []
    comparisons = {
        "cwru": ["LS-ProtoAlign", "DP-ProtoAug", "NoTransfer"],
        "wdbc": ["CORAL-FTL", "Entropy-ST", "NoTransfer"],
    }
    for ds, names in comparisons.items():
        runs = audit["datasets"][ds]["runs"]
        ours = np.asarray([r["FedKAPT (Ours)"]["accuracy"] for r in runs])
        local = []
        for name in names:
            base = np.asarray([r[name]["accuracy"] for r in runs])
            diff = ours - base
            mean, half = ci(diff)
            p = float(ttest_rel(ours, base).pvalue)
            dz = mean / diff.std(ddof=1) if diff.std(ddof=1) else 0.0
            local.append([ds, name, mean, half, p, dz,
                          float(np.mean(ours > base))])
        adj = holm([r[4] for r in local])
        for r, pa in zip(local, adj):
            r.append(float(pa)); rows.append(r)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired accuracy inference over ten seeds fixed in advance. "
        r"Differences are FedKAPT minus comparator; $d_z$ is the paired "
        r"standardised effect, and $p_{\rm H}$ is Holm-adjusted within dataset.}",
        r"\label{tab:paired}",
        r"\small",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Dataset & Comparator & Mean diff. [pp] & 95\% CI [pp] & $p$ & $p_{\rm H}$ & $d_z$ / win rate \\",
        r"\midrule",
    ]
    for ds, name, mean, half, p, dz, win, pa in rows:
        if ds == "wdbc" and name == comparisons["wdbc"][0]:
            lines.append(r"\midrule")
        lines.append(
            f"{LABELS[ds]} & {name} & {100*mean:+.2f} & "
            f"[{100*(mean-half):+.2f}, {100*(mean+half):+.2f}] & "
            f"{p:.3g} & {pa:.3g} & {dz:+.2f} / {100*win:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (PAPER / "paired_inference.tex").write_text("\n".join(lines),
                                                 encoding="utf-8")


def component_figure():
    setup(single_column=False)
    path = ART / "component_runs.json"
    data = json.loads(path.read_text())
    variants = list(data["summary"]["cwru"].keys())
    display = {
        "FedKAPT-Gen": "Gen",
        "FedKAPT-Core (condition only)": "FedKAPT",
        "FedKAPT": "FedKAPT",
        "w/o latent cycle": "No cycle",
        "w/o generator Sinkhorn": "No gen. OT",
        "w/o soft conditioning": "Hard cond.",
        "w/o filtering": "No filter",
        "w/o alignment head": "No align",
        "w/o late fusion": "No fusion",
    }
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 5.45))
    for row, ds in enumerate(("cwru", "wdbc")):
        base = data["summary"][ds]["FedKAPT-Gen"]
        for col, (metric, label) in enumerate(METRICS):
            ax = axes[row, col]
            means, errs = [], []
            for variant in variants:
                vals = np.asarray(
                    data["summary"][ds][variant][metric]["values"], float)
                full = np.asarray(base[metric]["values"], float)
                delta = 100 * (vals - full)
                m, e = ci(delta); means.append(m); errs.append(e)
            y = np.arange(len(variants))
            colors = [FEDKAPT_COLOR if v in ("FedKAPT-Core (condition only)", "FedKAPT")
                      else OTHER_COLORS[i % len(OTHER_COLORS)]
                      for i, v in enumerate(variants)]
            ax.barh(y, means, xerr=errs, color=colors, alpha=.82,
                    error_kw={"lw": .8, "capsize": 2})
            ax.axvline(0, color=".35", lw=.8)
            ax.set_yticks(y, [display[v] for v in variants]
                          if col == 0 else [])
            ax.invert_yaxis()
            ax.set_xlabel(f"$\\Delta$ {label} [pp]")
            panel(ax, chr(ord("a") + row * 4 + col))

        ax = axes[row, 3]
        gen = [r["FedKAPT-Gen"]["diagnostics"]
               for r in data["runs"][ds]]
        within = np.asarray(
            [d["sample_specificity_ratio"] for d in gen], float)
        cycle = np.asarray([d["latent_cycle_mse"] for d in gen], float)
        ax.scatter(100 * within, cycle, color=FEDKAPT_COLOR, s=25, alpha=.8)
        for idx, (x, yv) in enumerate(zip(100 * within, cycle)):
            ax.annotate(str(idx + 1), (x, yv), xytext=(3, 2),
                        textcoords="offset points", fontsize=6.5)
        ax.set_xlim(left=0)
        ax.set_xlabel("Generated/target dispersion [%]")
        ax.set_ylabel("Latent-cycle MSE [squared units]")
        panel(ax, chr(ord("a") + row * 4 + 3))

    for ax in axes.flat:
        ax.grid(True, color="#dddddd", lw=.5, alpha=.7, axis="x")
        ax.spines[["top", "right"]].set_visible(False)
    for idx in (0, 3, 4, 7):
        plt.setp(axes.flat[idx].get_yticklabels(), rotation=60, ha="right",
                 rotation_mode="anchor")
    for idx in (0, 4):
        plt.setp(axes.flat[idx].get_yticklabels(), fontsize=6.8)
    fig.subplots_adjust(left=.13, right=.99, bottom=.10, top=.88,
                        wspace=.22, hspace=.50)
    for row, label in enumerate(("CWRU", "WDBC")):
        positions = [axes[row, col].get_position() for col in range(4)]
        center = (min(pos.x0 for pos in positions)
                  + max(pos.x1 for pos in positions)) / 2
        top = max(pos.y1 for pos in positions)
        fig.text(center, top + .035, label, ha="center", va="bottom",
                 fontsize=mpl.rcParams["axes.titlesize"] + 1.0,
                 fontweight="bold")
    fig.savefig(FIG / "component_ablation_composite.pdf",
                bbox_inches="tight")
    plt.close(fig)


def scope_summary():
    data = json.loads((ART / "scope_runs.json").read_text())
    lines = []
    for ds, pairs in data["runs"].items():
        pair_rows = []
        for tag, seed_rows in pairs.items():
            vals = [(v.get("FedKAPT") or v["FedKAPT-Core"])["accuracy"]
                    for v in seed_rows.values()]
            bases = [v["Baseline"]["accuracy"] for v in seed_rows.values()]
            if not vals:
                continue
            pair_rows.append((tag, float(np.mean(vals)),
                              float(np.mean(np.asarray(vals)-np.asarray(bases))),
                              len(vals)))
        lines.append({
            "dataset": ds,
            "pairs": len(pair_rows),
            "runs": int(sum(r[3] for r in pair_rows)),
            "mean_accuracy": float(np.mean([r[1] for r in pair_rows])),
            "mean_gain": float(np.mean([r[2] for r in pair_rows])),
            "pair_results": [
                {"direction": r[0], "accuracy": r[1],
                 "gain_vs_target_only": r[2], "n": r[3]}
                for r in pair_rows],
        })
    (ART / "scope_summary.json").write_text(json.dumps(lines, indent=2),
                                            encoding="utf-8")
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Three-seed FedKAPT scope validation. Accuracy and "
        r"paired gain over the target-only classifier are percentage points.}",
        r"\label{tab:scope}",
        r"\small",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Dataset & Direction & Accuracy & Gain \\",
        r"\midrule",
    ]
    cw = lines[0]["pair_results"]
    oh = lines[1]["pair_results"]
    def append_group(dsname, rows):
        for i, row in enumerate(rows):
            direction = row["direction"].replace("_", r"\_").replace(
                "->", r"$\rightarrow$")
            tex.append(" & ".join([
                dsname if i == 0 else "", direction,
                f"{100*row['accuracy']:.2f}",
                f"{100*row['gain_vs_target_only']:+.2f}"]) + r" \\")
    append_group("CWRU", cw)
    tex.append(r"\midrule")
    append_group("WDBC", oh)
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER / "scope_results.tex").write_text("\n".join(tex), encoding="utf-8")
    for row in lines:
        print("SCOPE", row["dataset"], row["pairs"], row["runs"],
              f"acc={row['mean_accuracy']:.4f}",
              f"gain={row['mean_gain']:+.4f}")


def identifiability_figure():
    setup(single_column=True)
    data = json.loads((ART / "identifiability_study.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 3.42))
    axes = axes.ravel()
    perturbation_counts = data["protocol"].get(
        "perturbation_counts", [data["protocol"].get("perturbations", 50)])
    primary_B = int(data["protocol"].get(
        "primary_perturbations", max(perturbation_counts)))
    datasets = ("cwru", "wdbc")

    # (a) Post-hoc semantic assignment accuracy at the primary perturbation
    # count.  Labels are used for evaluation only and never by the screen.
    ax = axes[0]
    assignment = [[r["assignment_accuracy"]
                   for r in data["reliability"][ds]
                   if r.get("perturbations", primary_B) == primary_B]
                  for ds in datasets]
    bp = ax.boxplot(assignment, positions=[0, 1], widths=.48,
                    patch_artist=True, showfliers=False)
    for box, ds in zip(bp["boxes"], datasets):
        box.set(facecolor=COLORS[ds], edgecolor=COLORS[ds], alpha=.38)
    for idx, (values, ds) in enumerate(zip(assignment, datasets)):
        ax.scatter(np.full(len(values), idx), values, color=COLORS[ds],
                   s=8, alpha=.8, zorder=3)
    ax.set_xticks([0, 1], ["CWRU", "WDBC"])
    ax.set_ylabel("Assign. acc. [fraction]")
    ax.set_ylim(0, 1.05)
    panel(ax, "a")

    # (b) The two quantities used by the sufficient local stability decision.
    ax = axes[1]
    radius_means, radius_errs, eta_means, eta_errs = [], [], [], []
    for ds in datasets:
        rows = [r for r in data["reliability"][ds]
                if r.get("perturbations", primary_B) == primary_B]
        m, e = ci([r["stability_radius"] for r in rows])
        radius_means.append(m); radius_errs.append(e)
        m, e = ci([r["eta95"] for r in rows])
        eta_means.append(m); eta_errs.append(e)
    x = np.arange(2)
    ax.errorbar(x, radius_means, yerr=radius_errs,
                color="#4D4D4D", marker="s", capsize=2,
                label=r"Radius $\Gamma/(2K)$")
    ax.errorbar(x, eta_means, yerr=eta_errs,
                color=FEDKAPT_COLOR, marker="o", capsize=2,
                label=r"Local $\widehat\eta_{.95}$")
    ax.set_yscale("log")
    ax.set_xticks(x, ["CWRU", "WDBC"])
    ax.set_ylabel("Local stability [score]")
    ax.legend(frameon=False, loc="best", fontsize=5.5,
              handlelength=1.2, borderaxespad=.2)
    panel(ax, "b")

    # (c) Acceptance-rate sensitivity to the Monte Carlo sample count.
    ax = axes[2]
    for ds in datasets:
        rates = []
        for B in perturbation_counts:
            rows = [r for r in data["reliability"][ds]
                    if r.get("perturbations", B) == B]
            rates.append(float(np.mean([r["accepted"] for r in rows])))
        ax.plot(perturbation_counts, rates, marker="o", ms=3,
                color=COLORS[ds], label=LABELS[ds])
    ax.set_xticks(perturbation_counts)
    ax.set_xlabel("Perturbations $B$ [count]")
    ax.set_ylabel("Accept rate [fraction]")
    ax.set_ylim(-.05, 1.05)
    ax.legend(frameon=False, loc="best", fontsize=5.5,
              handlelength=1.2, borderaxespad=.2)
    panel(ax, "c")

    ax = axes[3]
    for ds in datasets:
        rows = [r for r in data["reliability"][ds]
                if r.get("perturbations", primary_B) == primary_B]
        x = np.asarray([r["stability_radius"] for r in rows])
        y = np.asarray([r["eta95"] for r in rows])
        ax.scatter(x, y, color=COLORS[ds], marker="o", s=13,
                   label=LABELS[ds], alpha=.85)
    lo, hi = 1e-7, 1.0
    ax.plot([lo, hi], [lo, hi], color="#4D4D4D", ls="--", lw=.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Stability radius $\Gamma/(2K)$ [score]")
    ax.set_ylabel(r"Perturbation $\widehat\eta_{.95}$ [score]")
    panel(ax, "d")

    ax = axes[4]
    stability = [[r["exact_assignment_stability"]
                  for r in data["reliability"][ds]
                  if r.get("perturbations", primary_B) == primary_B]
                 for ds in datasets]
    bp = ax.boxplot(stability, positions=[0, 1], widths=.48,
                    patch_artist=True, showfliers=False)
    for box, ds in zip(bp["boxes"], ("cwru", "wdbc")):
        box.set(facecolor=COLORS[ds], edgecolor=COLORS[ds], alpha=.38)
    for idx, (values, ds) in enumerate(zip(
            stability, datasets)):
        ax.scatter(np.full(len(values), idx), values, color=COLORS[ds],
                   s=10, alpha=.8, zorder=3)
    ax.set_xticks([0, 1], ["CWRU", "WDBC"])
    ax.set_ylabel("Exact stability [fraction]")
    ax.set_ylim(-.05, 1.05)
    panel(ax, "e")

    ax = axes[5]
    controls = ["Transport-Label-Only", "One-Hot-Condition",
                "DP-Prototype-Condition"]
    display = ["Label only", "One-hot", "DP prototype"]
    palette = [OTHER_COLORS[0], OTHER_COLORS[1], FEDKAPT_COLOR]
    x = np.arange(2); width = .23
    for idx, (name, label, color) in enumerate(zip(controls, display, palette)):
        means, errors = [], []
        for ds in datasets:
            vals = [r["metrics"][name]["accuracy"]
                    for r in data["condition_controls"][ds]]
            m, e = ci(vals); means.append(m); errors.append(e)
        ax.bar(x + (idx - 1) * width, means, width, yerr=errors,
               color=color, alpha=.86, capsize=2, label=label)
    ax.set_xticks(x, ["CWRU", "WDBC"])
    ax.set_ylabel("Target acc. [fraction]")
    panel(ax, "f")

    for ax in axes:
        ax.set_box_aspect(1)
        ax.grid(True, color="#dddddd", lw=.5, alpha=.72)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        mpl.lines.Line2D([], [], color=COLORS["cwru"], marker="o",
                         label=LABELS["cwru"]),
        mpl.lines.Line2D([], [], color=COLORS["wdbc"], marker="o",
                         label=LABELS["wdbc"]),
    ] + [mpl.patches.Patch(color=color, alpha=.86, label=label)
         for label, color in zip(display, palette)]
    dataset_legend = fig.legend(
        handles=handles[:2], loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(.5, .975), columnspacing=1.0, handlelength=1.5)
    fig.add_artist(dataset_legend)
    fig.legend(handles=handles[2:], loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(.5, .92), columnspacing=.65,
               handlelength=1.2)
    fig.subplots_adjust(left=.12, right=.99, bottom=.10, top=.81,
                        wspace=.58, hspace=.22)
    fig.savefig(FIG / "identifiability_evidence_composite.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main():
    setup()
    structural_figure()
    component_figure()
    identifiability_figure()
    if PAPER.is_dir():
        PAPER_FIG.mkdir(parents=True, exist_ok=True)
        for name in ("structural_sensitivity_composite.pdf",
                     "component_ablation_composite.pdf",
                     "identifiability_evidence_composite.pdf"):
            shutil.copy2(FIG / name, PAPER_FIG / name)
    print(FIG / "structural_sensitivity_composite.pdf")


if __name__ == "__main__":
    main()
