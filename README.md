# terc-rl

> **Note:** This README is a placeholder. The full description, diagram, and
> results figures will be added last.

**TERC** — a Transfer Entropy Redundancy Criterion for state-variable selection
in reinforcement learning. Given trajectories from a trained agent, TERC
identifies the smallest subset of observable state variables that the agent's
actions actually depend on, correctly resolving redundant and synergistic
relationships that pairwise feature-selection methods cannot.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
```

The `LunarLander` environment additionally needs Box2D (requires SWIG and a
C/C++ toolchain): `pip install gymnasium[box2d]` (or `pip install -e ".[lunarlander]"`).
All other environments work without it.

## Quick start

```bash
# Full pipeline: train -> select variables -> learning curves
python main.py --name CartPole --stage all

# Synthetic redundancy datasets
python main.py --name 4red_varbs --stage all
python main.py --name 2red_trips --stage all

# TERC selection only, per-training-quartile interpretability analysis
python main.py --name CartPole --stage terc --quartiles
```

Supported `--name` values:

| Category  | Names |
|-----------|-------|
| Synthetic | `4red_varbs`, `2red_trips` |
| Custom RL | `SKG` (Secret Key Game), `TFMT` (Tit-For-N-Tats sweep) |
| Gym       | `CartPole`, `LunarLander`, `Pendulum` |

## Package layout

```
terc/
  estimator.py            TERC criterion + MINE neural TE estimator
  agents/                 Actor-Critic, PPO, tabular Q-learning
  envs/                   Secret Key Game, Tit-For-N-Tats, Gym helpers
  pipelines/              synthetic data, trajectory & learning-curve generation
  visualization.py        plotting helpers
baselines/seek/           SEEK-style knockoff baseline on the synthetic datasets
main.py                   command-line entry point
tests/test_smoke.py       fast end-to-end smoke tests
```

## Tests

```bash
PYTHONPATH=. python tests/test_smoke.py     # or: pytest tests/
```
