"""
core/ot_alignment.py — Module 2: Partial OT Semantic Alignment.
"""

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from config import Config


def _cost_matrix(proto_t, proto_d, metric):
    if metric == "cosine":
        sim = normalize(proto_t, "l2") @ normalize(proto_d, "l2").T
        M   = (1.0 - sim).clip(0)
    elif metric == "euclidean":
        diff = proto_t[:, None, :] - proto_d[None, :, :]
        M = np.sqrt((diff ** 2).sum(-1))
    else:
        diff = proto_t[:, None, :] - proto_d[None, :, :]
        M    = (diff ** 2).sum(-1)
    return (M / (M.max() + 1e-8)).astype(np.float64)


def _intra_cost(proto, metric):
    if metric == "cosine":
        sim = normalize(proto, "l2") @ normalize(proto, "l2").T
        M = (1.0 - sim).clip(0)
    elif metric == "euclidean":
        diff = proto[:, None, :] - proto[None, :, :]
        M = np.sqrt((diff ** 2).sum(-1))
    else:
        diff = proto[:, None, :] - proto[None, :, :]
        M = (diff ** 2).sum(-1)
    return (M / (M.max() + 1e-8)).astype(np.float64)


def _quantile_signatures(C, q_dim):
    """Fixed-length signatures valid for unequal cluster/class counts."""
    q = np.linspace(0.0, 1.0, q_dim)
    signatures = []
    for idx, row in enumerate(C):
        values = np.delete(row, idx) if len(row) > 1 else row
        signatures.append(np.quantile(values, q))
    return np.asarray(signatures, dtype=np.float64)


def _relational_cost(proto_t, proto_d, metric, signature_dim=32):
    Ct = _intra_cost(proto_t, metric)
    Cd = _intra_cost(proto_d, metric)
    sig_t = _quantile_signatures(Ct, signature_dim)
    sig_d = _quantile_signatures(Cd, signature_dim)
    diff = sig_t[:, None, :] - sig_d[None, :, :]
    M = (diff ** 2).sum(-1)
    return (M / (M.max() + 1e-8)).astype(np.float64)


def _solve_partial(a, b, M, mass, reg, n_iter):
    try:
        return ot.partial.entropic_partial_wasserstein(
            a, b, M, reg=max(reg, 1e-4), m=mass,
            numItermax=n_iter, stopThr=1e-8)
    except Exception:
        return ot.partial.partial_wasserstein(a, b, M, m=mass)


def constrained_bijection(score, anchors=None):
    """Maximise a square assignment score subject to disclosed anchors."""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("Constrained bijection requires a square score matrix")
    n = score.shape[0]
    fixed = {int(row): int(col) for row, col in (anchors or [])}
    if any(row < 0 or row >= n or col < 0 or col >= n
           for row, col in fixed.items()):
        raise ValueError("Anchor index is outside the assignment matrix")
    if len(set(fixed.values())) != len(fixed):
        raise ValueError("Anchors must map to distinct source classes")
    free_rows = [row for row in range(n) if row not in fixed]
    free_cols = [col for col in range(n) if col not in set(fixed.values())]
    mapping = np.full(n, -1, dtype=np.int64)
    for row, col in fixed.items():
        mapping[row] = col
    if free_rows:
        rows, cols = linear_sum_assignment(-score[np.ix_(free_rows, free_cols)])
        for row, col in zip(rows, cols):
            mapping[free_rows[row]] = free_cols[col]
    return mapping


def _best_mass(proto_t, proto_d, M, grid, reg=0.05, n_iter=200):
    C, K = M.shape
    a = np.ones(C) / C
    b = np.ones(K) / K
    best_s, best_score = 1.0, -np.inf
    for s in grid:
        try:
            T     = _solve_partial(a, b, M, s, reg, n_iter)
            T_n   = T / (T.sum(axis=1, keepdims=True) + 1e-10)
            score = float(T_n.max(axis=1).mean())
            if score > best_score:
                best_score, best_s = score, s
        except Exception:
            continue
    return best_s


class PartialOTAligner:
    def __init__(self, cfg: Config):
        self.cfg              = cfg
        self.T_star           = None
        self.cluster_to_class = None
        self.mass             = None

    def fit(self, proto_t, proto_d) -> "PartialOTAligner":
        cfg_ot = self.cfg.ot
        C, K   = len(proto_t), len(proto_d)
        # Store t-side prototypes for ProtoFTL-style conditioning (use_proto_condition).
        self.proto_t = proto_t.astype(np.float32)
        a      = np.ones(C) / C
        b      = np.ones(K) / K

        # Always use the relational (structural) cost for OT so that T_soft
        # captures within-domain class geometry rather than cross-domain
        # prototype distances.  This is critical for two reasons:
        #   1. OC features are disjoint CNN descriptor halves — direct
        #      cross-half cosine distance is semantically meaningless.
        #   2. DP noise perturbs individual prototype coordinates, making
        #      direct cross-domain distances unreliable even for CWRU.
        # Relational cost only depends on WITHIN-domain pairwise distances,
        # which are far more robust to DP perturbations.
        M_rel = _relational_cost(
            proto_t, proto_d, cfg_ot.cost_metric,
            getattr(cfg_ot, "signature_dim", 32))
        direct_w = float(getattr(cfg_ot, "direct_cost_weight", 0.0))
        if direct_w > 0.0 and proto_t.shape[1] == proto_d.shape[1]:
            M_direct = _cost_matrix(proto_t, proto_d, cfg_ot.cost_metric)
            direct_w = min(max(direct_w, 0.0), 1.0)
            M_ot = (1.0 - direct_w) * M_rel + direct_w * M_direct
        else:
            M_ot = M_rel
        self.relational_cost = M_ot.astype(np.float32)

        self.mass = cfg_ot.partial_mass or _best_mass(
            proto_t, proto_d, M_ot, cfg_ot.mass_search_grid,
            cfg_ot.sinkhorn_reg, cfg_ot.n_sinkhorn_iter)

        T = _solve_partial(
            a, b, M_ot, self.mass, cfg_ot.sinkhorn_reg,
            cfg_ot.n_sinkhorn_iter)
        # T_soft: raw Sinkhorn output used for soft OT-mixture conditioning.
        # T_star: may be overwritten by bijection in _cluster_mapping (for heatmap).
        self.T_soft = T.astype(np.float32)
        self.T_star = T.astype(np.float32)
        self.cluster_to_class = self._cluster_mapping(self.T_star)
        return self

    def _cluster_mapping(self, T):
        T_n = T / (T.sum(axis=1, keepdims=True) + 1e-10)
        C, K = T_n.shape
        # FIX: relational_cost_weight read from config (default 0.25, was hardcoded 0.50).
        # Reducing from 0.50 to 0.25 avoids over-penalising structural mismatch when
        # prototypes are noisy from DP perturbation, leading to better assignments.
        if C == K:
            # C==K: one-to-one bipartite assignment prevents pseudo-label collapse.
            # The hard correspondence is extracted from the same partial plan
            # that supplies the soft conditioning weights. Optional disclosed
            # anchors constrain only the remaining one-to-one assignment.
            score = T_n
            mapping = constrained_bijection(
                score, getattr(self.cfg.ot, "public_anchors", None))
            T_assign = np.zeros_like(T_n, dtype=np.float32)
            T_assign[np.arange(C), mapping] = 1.0
            self.T_star = T_assign / max(C, 1)
            return mapping
        # For C != K, map each target cluster through the relational plan.
        # This avoids any direct coordinate comparison between heterogeneous
        # spaces and remains defined for arbitrary C and K.
        return T_n.argmax(axis=1).astype(np.int64)

    def compute_transport_conditions(self, proto_d_noisy, cluster_assignments,
                                     hard: bool = False, uniform: bool = False,
                                     sample_features=None):
        """
        Compute per-sample transport conditions.

        uniform=True (ablation "w/o Partial OT"):
            No OT alignment — every sample receives the plain average of all
            d-prototypes.  Tests whether OT-guided alignment contributes.

        hard=True (ablation "w/o Soft Cond."):
            Hard assignment — each sample is conditioned on the single
            d-prototype selected by hard argmax.
            When use_proto_condition=True: argmax of t-prototype similarity.
            When use_proto_condition=False: OT bijection (T_star / LAP).

        hard=False, uniform=False (default, Full FedKAPT):
            When use_proto_condition=True (OC):
                ProtoFTL-style soft conditions — temperature-scaled softmax
                of t-prototype similarity weights d-prototypes reordered by
                the OT bijection.  Reliable for OC where cross-domain OT
                conditions are unreliable (disjoint CNN feature halves).
                Yields smooth, well-calibrated features → better AUC.
            When use_proto_condition=False (CWRU, default):
                Soft OT transport plan (T_soft) — each cluster's condition is
                a convex combination of d-prototypes weighted by the continuous
                Sinkhorn partial-transport plan.
        """
        if uniform:
            # No alignment: average all d-prototypes uniformly
            avg = proto_d_noisy.mean(axis=0, keepdims=True)
            return np.tile(avg, (len(cluster_assignments), 1)).astype(np.float32)

        use_proto = getattr(self.cfg.ot, "use_proto_condition", False)
        if use_proto and sample_features is not None and hasattr(self, "proto_t"):
            # ProtoFTL-style: t-side prototype similarity → weighted d-prototype mix.
            # Uses t-space similarities (reliable, same feature space) instead of
            # cross-domain OT distances (unreliable when features are disjoint).
            # Reorders d-prototypes by OT bijection so cluster k → d_class k.
            t_n   = normalize(sample_features, "l2")            # [N, t_dim]
            mu_t  = normalize(self.proto_t,    "l2")            # [C, t_dim]
            sim   = t_n @ mu_t.T                                # [N, C] raw cosine
            d_ord = proto_d_noisy[self.cluster_to_class]        # [C, d_dim]
            if hard:
                # Hard: nearest t-prototype → single d-prototype (one-hot)
                nn_idx = sim.argmax(axis=1)                     # [N]
                return d_ord[nn_idx].astype(np.float32)
            # Soft: temperature-scaled softmax — smooth convex combination
            temp  = float(getattr(self.cfg.ot, "proto_cond_temp", 5.0))
            s     = sim * temp
            s    -= s.max(axis=1, keepdims=True)                # numerical stability
            w     = np.exp(s)
            w    /= w.sum(axis=1, keepdims=True) + 1e-10        # [N, C]
            return (w @ d_ord).astype(np.float32)

        if hard:
            # Hard bijection (T_star / LAP): one d-prototype per cluster.
            # Equivalent to one-hot W @ proto_d, implemented as direct index.
            return proto_d_noisy[self.cluster_to_class[cluster_assignments]].astype(
                np.float32)

        # Default Full FedKAPT: use T_soft (continuous Sinkhorn plan).
        # T_soft is set in fit() from the raw partial_wasserstein output and
        # is NEVER overwritten by the bijection step (unlike T_star).
        # Row-normalise to obtain a valid convex combination per cluster.
        T_for_cond = self.T_soft
        W = T_for_cond[cluster_assignments]
        W = W / (W.sum(axis=1, keepdims=True) + 1e-10)
        ot_cond = (W @ proto_d_noisy).astype(np.float32)
        nn_w = float(getattr(self.cfg.ot, "nn_condition_weight", 0.0))
        if nn_w > 0.0 and sample_features is not None:
            nn_cond = self._sample_nn_conditions(proto_d_noisy, sample_features)
            nn_w = min(max(nn_w, 0.0), 1.0)
            return ((1.0 - nn_w) * ot_cond + nn_w * nn_cond).astype(np.float32)
        return ot_cond

    def _sample_nn_conditions(self, proto_d_noisy, sample_features):
        align_dim = min(sample_features.shape[1], proto_d_noisy.shape[1])
        X_n = normalize(sample_features[:, :align_dim], norm="l2")
        D_n = normalize(proto_d_noisy[:, :align_dim], norm="l2")
        nn_idx = (X_n @ D_n.T).argmax(axis=1)
        return proto_d_noisy[nn_idx].astype(np.float32)

    def get_pseudo_labels(self, cluster_assignments):
        return self.cluster_to_class[cluster_assignments]

    def alignment_summary(self) -> dict:
        T_n     = self.T_star / (self.T_star.sum(1, keepdims=True) + 1e-10)
        entropy = -(T_n * np.log(T_n + 1e-10)).sum(1)
        return {
            "transport_mass":     self.mass,
            "cluster_to_class":   self.cluster_to_class.tolist(),
            "T_star_shape":       list(self.T_star.shape),
            "T_star_row_entropy": float(entropy.mean()),
        }
