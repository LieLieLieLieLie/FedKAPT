"""Focused regression tests for issues raised during the paper review."""

import numpy as np
import pytest
from scipy.stats import norm

from config import Config, DataConfig, OTConfig
from core.ot_alignment import PartialOTAligner, constrained_bijection
from core.privacy import compute_sigma
from trainer import FedKAPTTrainer


def test_analytic_gaussian_profile_matches_delta():
    epsilon, delta, sensitivity = 10.0, 1e-5, 0.04
    sigma = float(compute_sigma(sensitivity, epsilon, delta))
    mu = sensitivity / sigma
    achieved = (norm.cdf(-epsilon / mu + mu / 2.0)
                - np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2.0))
    assert np.isclose(achieved, delta, rtol=1e-7, atol=1e-12)


def test_relational_alignment_accepts_unequal_cardinality():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(7, 13)).astype(np.float32)
    source = rng.normal(size=(10, 5)).astype(np.float32)
    cfg = Config(data=DataConfig(n_classes=10),
                 ot=OTConfig(signature_dim=16, direct_cost_weight=0.0))
    aligner = PartialOTAligner(cfg).fit(target, source)
    assert aligner.T_soft.shape == (7, 10)
    assert aligner.cluster_to_class.shape == (7,)
    assert np.all((0 <= aligner.cluster_to_class)
                  & (aligner.cluster_to_class < 10))


def test_declared_anchors_constrain_remaining_bijection():
    score = np.array([
        [9.0, 8.0, 0.0],
        [8.0, 7.0, 6.0],
        [0.0, 6.0, 5.0],
    ])
    mapping = constrained_bijection(score, anchors=[[0, 1]])
    assert mapping.tolist() == [1, 0, 2]
    assert sorted(mapping.tolist()) == [0, 1, 2]


def test_alignment_cannot_bypass_private_release():
    cfg = Config(ot=OTConfig(clean_alignment=True))
    trainer = FedKAPTTrainer(cfg)
    with pytest.raises(ValueError, match="one-message DP contract"):
        trainer.phase2_ot_alignment()
