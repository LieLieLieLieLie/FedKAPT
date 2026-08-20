"""Wisconsin Diagnostic Breast Cancer data for feature-disjoint FTL.

The source party observes the ten mean cell-nucleus descriptors, whereas the
target party observes the corresponding ten worst-value descriptors on a
disjoint patient cohort.  Each party standardises its own descriptors using
training-partition statistics only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import Config


CLASS_NAMES = ["Benign", "Malignant"]


class WDBCDataModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = os.path.join(cfg.data.wdbc_data_dir, "wdbc.data")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"WDBC data file not found: {path}")

        frame = pd.read_csv(path, header=None)
        x = frame.iloc[:, 2:32].to_numpy(dtype=np.float32)
        y = (frame.iloc[:, 1].astype(str).str.strip() == "M").to_numpy(
            dtype=np.int64)
        all_idx = np.arange(len(y))
        d_idx, t_idx = train_test_split(
            all_idx,
            test_size=0.5,
            random_state=cfg.data.wdbc_partition_seed,
            stratify=y,
        )

        # Ten semantically corresponding but nonidentical coordinate systems:
        # mean descriptors at d versus worst-value descriptors at t.
        d_x, d_y = x[d_idx, :10], y[d_idx]
        t_x, t_y = x[t_idx, 20:30], y[t_idx]

        (t_train_x, t_test_x,
         self.t_train_y, self.t_test_y) = train_test_split(
            t_x,
            t_y,
            test_size=cfg.data.test_ratio,
            random_state=cfg.seed,
            stratify=t_y,
        )
        (d_train_x, d_test_x,
         self.d_train_y, self.d_test_y) = train_test_split(
            d_x,
            d_y,
            test_size=cfg.data.test_ratio,
            random_state=cfg.seed,
            stratify=d_y,
        )

        t_scaler = StandardScaler().fit(t_train_x)
        d_scaler = StandardScaler().fit(d_train_x)
        self.t_train_x = t_scaler.transform(t_train_x).astype(np.float32)
        self.t_test_x = t_scaler.transform(t_test_x).astype(np.float32)
        self.d_train_x = d_scaler.transform(d_train_x).astype(np.float32)
        self.d_test_x = d_scaler.transform(d_test_x).astype(np.float32)
        self.classes = list(CLASS_NAMES)
        cfg.data.n_classes = len(self.classes)

    def summary(self) -> dict:
        return {
            "dataset": "WDBC",
            "source (d-side)": "mean nucleus descriptors",
            "target (t-side)": "worst-value nucleus descriptors",
            "classes": len(self.classes),
            "class_names": self.classes,
            "t_train": len(self.t_train_x),
            "t_test": len(self.t_test_x),
            "d_train": len(self.d_train_x),
            "d_test": len(self.d_test_x),
            "t_feat_dim": self.t_train_x.shape[1],
            "d_feat_dim": self.d_train_x.shape[1],
        }
