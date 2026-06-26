"""TERC: A Transfer Entropy Redundancy Criterion for State Variable Selection
in Reinforcement Learning.

Public API:

    from terc import TERC, TrajectoryGenerator, LearningCurveGenerator, generate_data
"""

from terc.estimator import TERC, MineNet
from terc.pipelines import TrajectoryGenerator, LearningCurveGenerator, generate_data

__version__ = "0.1.0"

__all__ = [
    "TERC",
    "MineNet",
    "TrajectoryGenerator",
    "LearningCurveGenerator",
    "generate_data",
]
