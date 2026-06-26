"""Fast smoke tests for the TERC package.

These run tiny configurations (a handful of trajectories / MINE iterations) to
verify that every component imports, runs, and produces outputs of the expected
shape. They are NOT a reproduction of the paper's results.

Run with either::

    pytest tests/
    python tests/test_smoke.py
"""

import tempfile
from pathlib import Path

import numpy as np

from terc import TERC, TrajectoryGenerator, LearningCurveGenerator, generate_data
from terc.envs import SecretKeyGameEnv, TitForNTatsEnv, make_gym_env


def test_synthetic_generation_and_terc():
    with tempfile.TemporaryDirectory() as d:
        obs, acs = generate_data("4red_varbs", n_points=200, data_dir=d)
        assert obs.shape == (200, 6)
        assert set(np.unique(acs)).issubset({0, 1})
        keepers = TERC(name="4red_varbs", num_iterations=20, batch_size=16,
                       n_experiments=2, data_dir=d, results_dir=d,
                       verbose=False).run()
        assert isinstance(keepers, list)
        assert (Path(d) / "4red_varbs_means.npy").exists()


def test_secret_key_game_env():
    env = SecretKeyGameEnv(state_length=25)
    state = env.reset()[0]
    assert state.shape == (25,)
    assert env.action_space.n == 80
    s, r, term, trunc, info = env.step(40)
    assert s.shape == (25,)
    assert isinstance(r, int) or np.isscalar(r)


def test_tit_for_n_tats_env():
    env = TitForNTatsEnv(3, max_mem_size=9)
    state = env.reset()[0]
    assert len(state) == 9
    s, r, term, trunc, info = env.step(1)
    assert len(s) == 9


def test_skg_trajectory_and_terc():
    with tempfile.TemporaryDirectory() as d:
        TrajectoryGenerator(name="SKG", num_trajectories=30, data_dir=d,
                            skg_state_length=10, verbose=False).run()
        assert (Path(d) / "obs_SKG.npy").exists()
        keepers = TERC(name="SKG", num_iterations=20, batch_size=16,
                       n_experiments=2, data_dir=d, results_dir=d,
                       verbose=False).run()
        assert isinstance(keepers, list)


def test_gym_cartpole_trajectory():
    with tempfile.TemporaryDirectory() as d:
        gen = TrajectoryGenerator(name="CartPole", num_trajectories=5, data_dir=d,
                                  verbose=False)
        gen.run()
        obs = np.load(Path(d) / "obs_CartPole.npy")
        # 4 CartPole state dims + 3 doping random variables
        assert obs.shape[1] == 7


def test_gym_env_factory():
    env = make_gym_env("CartPole")
    out = env.reset()
    assert len(out[0]) == 4


def test_learning_curve_smoke():
    with tempfile.TemporaryDirectory() as d:
        LearningCurveGenerator(name="CartPole", keep_tene=[0, 1, 2, 3],
                               num_trajectories=3, n_experiments=1,
                               data_dir=d, verbose=False).run()
        lcs = np.load(Path(d) / "learning_curve_CartPole.npy")
        assert lcs.shape[0] == 2  # full state and TERC subset


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    raise SystemExit(failed)
