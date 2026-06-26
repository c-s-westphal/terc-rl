#!/usr/bin/env python3
"""Command-line entry point for the TERC pipeline.

Runs the three TERC stages for a chosen environment / dataset:

    1. ``trajectories``  -- train an agent and sample state-action trajectories
    2. ``terc``          -- apply TERC to select the minimal state subset
    3. ``curves``        -- retrain on the full vs. selected state and record
                            learning curves

Examples
--------
    # Full pipeline on Cart Pole
    python main.py --name CartPole --stage all

    # Synthetic redundancy dataset (no RL learning curves)
    python main.py --name 4red_varbs --stage all

    # TERC only, per-training-quartile interpretability analysis
    python main.py --name CartPole --stage terc --quartiles

Supported ``--name`` values:
    Synthetic : 4red_varbs, 2red_trips
    Custom RL : SKG (Secret Key Game), TFMT (Tit-For-N-Tats sweep)
    Gym       : CartPole, LunarLander, Pendulum
"""

import argparse

from terc.envs import resolve_gym_name, GYM_ENVS
from terc.pipelines import TrajectoryGenerator, LearningCurveGenerator
from terc.estimator import TERC

SYNTHETIC = {"4red_varbs", "2red_trips"}


def build_parser():
    p = argparse.ArgumentParser(
        description="TERC: Transfer Entropy Redundancy Criterion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--name", required=True,
                   help="Environment / dataset name (e.g. CartPole, SKG, TFMT, 4red_varbs).")
    p.add_argument("--stage", default="all",
                   choices=["trajectories", "terc", "curves", "all"],
                   help="Pipeline stage(s) to run.")
    p.add_argument("--num-trajectories", type=int, default=10000,
                   help="Number of trajectories to roll out during generation.")
    p.add_argument("--num-iters", type=int, default=10,
                   help="Number of MINE training iterations for TERC.")
    p.add_argument("--batch-size", type=int, default=100, help="MINE mini-batch size.")
    p.add_argument("--lr", type=float, default=0.0001, help="MINE learning rate.")
    p.add_argument("--lra", type=float, default=0.0001, help="Actor learning rate.")
    p.add_argument("--lrc", type=float, default=0.001, help="Critic learning rate.")
    p.add_argument("--n-experiments", type=int, default=5,
                   help="Repeated MINE runs averaged per variable.")
    p.add_argument("--quartiles", action="store_true",
                   help="Run the per-training-quartile TERC analysis instead of "
                        "the standard full-trajectory selection.")
    p.add_argument("--curve-trajectories", type=int, default=3000,
                   help="Number of episodes for learning-curve generation.")
    p.add_argument("--data-dir", default="outputs",
                   help="Directory for trajectories and learning curves.")
    p.add_argument("--results-dir", default="outputs",
                   help="Directory for TERC MI curves.")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return p


def main():
    args = build_parser().parse_args()
    verbose = not args.quiet
    is_synthetic = args.name in SYNTHETIC
    is_gym = resolve_gym_name(args.name) in GYM_ENVS

    keepers = None
    run_traj = args.stage in ("trajectories", "all")
    run_terc = args.stage in ("terc", "all")
    run_curves = args.stage in ("curves", "all")

    if run_traj:
        TrajectoryGenerator(
            name=args.name, lra=args.lra, lrc=args.lrc,
            num_trajectories=args.num_trajectories,
            data_dir=args.data_dir, verbose=verbose).run()

    if run_terc:
        keepers = TERC(
            name=args.name, num_iterations=args.num_iters, full=not args.quartiles,
            batch_size=args.batch_size, lr=args.lr, n_experiments=args.n_experiments,
            data_dir=args.data_dir, results_dir=args.results_dir, verbose=verbose).run()
        print(f"TERC selected variable indices: {keepers}")

    if run_curves:
        if is_synthetic:
            print("Skipping learning curves: not applicable to synthetic datasets.")
        elif args.quartiles:
            print("Skipping learning curves: incompatible with --quartiles.")
        else:
            if keepers is None:
                raise SystemExit(
                    "Learning curves require TERC-selected indices. Run the "
                    "'terc' or 'all' stage first, or pass --stage all.")
            LearningCurveGenerator(
                name=args.name, keep_tene=keepers, lra=args.lra, lrc=args.lrc,
                num_trajectories=args.curve_trajectories,
                n_experiments=args.n_experiments, data_dir=args.data_dir,
                verbose=verbose).run()


if __name__ == "__main__":
    main()
