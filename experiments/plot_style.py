"""
experiments/plot_style.py — 全局绘图样式配置

颜色规范：
  FedKAPT        → #FF6666
  其他方法 (按序) → #FFAA53, #50CC55, #3399FF, #6666FF,
                         #9933FF, #00DDDD, #4D4D4D
  热力图(非负值)  → 0:#FFFFFF → max:#FF4F4F
  热力图(含正负)  → min:#007FFF → 0:#FFFFFF → max:#FF4F4F
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── 颜色 ─────────────────────────────────────────────────────────────────────

FEDKAPT_COLOR = "#FF6666"

OTHER_COLORS = [
    "#FFAA53",  # 0
    "#50CC55",  # 1
    "#3399FF",  # 2
    "#6666FF",  # 3
    "#9933FF",  # 4
    "#00DDDD",  # 5
    "#4D4D4D",  # 6
]

# 方法顺序（CrossFGAT-Lite → ProtoFTL）
_METHODS_IN_ORDER = [
    "NoTransfer",
    "FedAvg-FTL",
    "DANN-FTL",
    "SHOT-FTL",
    "ProtoFTL",
]


def method_color(name: str) -> str:
    if "FedKAPT" in name or "FedKAPT" in name or "Ours" in name:
        return FEDKAPT_COLOR
    try:
        idx = _METHODS_IN_ORDER.index(name)
        return OTHER_COLORS[idx % len(OTHER_COLORS)]
    except ValueError:
        return OTHER_COLORS[abs(hash(name)) % len(OTHER_COLORS)]


# t-SNE / 类别颜色（最多10类）
CLASS_COLORS = [
    "#FF6666", "#FFAA53", "#50CC55", "#00DDDD",
    "#3399FF", "#6666FF", "#9933FF", "#FF69B4",
    "#A0522D", "#2E8B57",
]

# ── 字号 ─────────────────────────────────────────────────────────────────────

FS_TICK   = 14
FS_LABEL  = 15
FS_TITLE  = 16
FS_LEGEND = 13
FS_ANNOT  = 12

# ── 热力图 colormap ───────────────────────────────────────────────────────────

def make_seq_cmap():
    """非负值: 0 的白色 → 最大正值的 #FF4F4F。"""
    return mcolors.LinearSegmentedColormap.from_list(
        "fedkapt_seq", ["#FFFFFF", "#FF4F4F"], N=256)


def make_div_cmap():
    """含正负值: 最大负值 #007FFF → 0 的白色 → 最大正值 #FF4F4F。"""
    return mcolors.LinearSegmentedColormap.from_list(
        "fedkapt_div", ["#007FFF", "#FFFFFF", "#FF4F4F"], N=256)


# ── 全局 rcParams ─────────────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update({
        "font.family":        "Times New Roman",
        "font.serif":         ["Times New Roman"],
        "mathtext.fontset":   "stix",
        "font.size":          FS_TICK,
        "axes.titlesize":     FS_TITLE,
        "axes.labelsize":     FS_LABEL,
        "xtick.labelsize":    FS_TICK,
        "ytick.labelsize":    FS_TICK,
        "legend.fontsize":    FS_LEGEND,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "grid.color":         "#CCCCCC",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "lines.linewidth":    2.5,
        "lines.markersize":   8,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         150,
        "savefig.dpi":        300,
    })


# ── 保存 ─────────────────────────────────────────────────────────────────────

def save_pdf(fig, path: str, tight: bool = True):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kw = dict(format="pdf", bbox_inches="tight") if tight else dict(format="pdf")
    fig.savefig(path, **kw)
    plt.close(fig)


# ── XLSX ─────────────────────────────────────────────────────────────────────

def save_xlsx(df_or_rows, path: str, sheet_name: str = "Sheet1",
              header=None, index: bool = False):
    try:
        import pandas as pd
        import openpyxl
    except ImportError:
        print(f"  [Style] openpyxl/pandas not installed; skipping XLSX: {path}")
        return

    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    df = pd.DataFrame(df_or_rows) if not isinstance(df_or_rows, pd.DataFrame) \
         else df_or_rows
    if header is not None:
        df.columns = header

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(
                (len(str(cell.value or "")) for cell in col), default=8)
            ws.column_dimensions[col[0].column_letter].width = max_len + 4
