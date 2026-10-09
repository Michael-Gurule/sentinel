"""Kalman filtering, gating, assignment, and multi-target tracking."""

from sentinel.tracking.assignment import assign_gnn
from sentinel.tracking.classes import pool_class_evidence
from sentinel.tracking.imm import IMMState, imm_predict, imm_update, transition_matrix
from sentinel.tracking.kalman import Gaussian, Innovation, innovation, predict, update
from sentinel.tracking.models import ConstantVelocity
from sentinel.tracking.tracker import (
    LinearMeasurement,
    Measurement,
    MultiTargetTracker,
    Track,
    TrackStatus,
)

__all__ = [
    "ConstantVelocity",
    "Gaussian",
    "IMMState",
    "Innovation",
    "LinearMeasurement",
    "Measurement",
    "MultiTargetTracker",
    "Track",
    "TrackStatus",
    "assign_gnn",
    "imm_predict",
    "imm_update",
    "innovation",
    "pool_class_evidence",
    "predict",
    "transition_matrix",
    "update",
]
