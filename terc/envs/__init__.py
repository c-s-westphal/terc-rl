"""Environments used in the TERC experiments.

This module provides the two custom environments (Secret Key Game and
Tit-For-N-Tats) as well as a small helper for constructing the Gymnasium
physics environments used in the paper, optionally "doped" with extra random
state variables (following Grooten et al., 2023).
"""

from terc.envs.secret_key_game import SecretKeyGameEnv
from terc.envs.tit_for_n_tats import TitForNTatsEnv

# Mapping from the friendly names used throughout the package / CLI to the
# Gymnasium environment id and the number of random "doping" variables added to
# the state in the paper's experiments.
GYM_ENVS = {
    "CartPole": {"id": "CartPole-v1", "n_random": 3},
    "LunarLander": {"id": "LunarLander-v3", "n_random": 3},
    "Pendulum": {"id": "Pendulum-v1", "n_random": 3},
}

# Aliases (including the legacy gym ids) for convenience.
_ALIASES = {
    "CartPole-v1": "CartPole",
    "LunarLander-v2": "LunarLander",
    "LunarLander-v3": "LunarLander",
    "Pendulum-v1": "Pendulum",
}


def resolve_gym_name(name):
    """Return the canonical friendly name for a Gymnasium environment."""
    return _ALIASES.get(name, name)


def make_gym_env(name):
    """Construct a Gymnasium environment from a friendly name.

    Falls back to the legacy ``LunarLander-v2`` id if ``v3`` is unavailable in
    the installed Gymnasium version.
    """
    import gymnasium as gym

    canonical = resolve_gym_name(name)
    if canonical not in GYM_ENVS:
        raise ValueError(f"Unknown Gym environment: {name!r}")
    env_id = GYM_ENVS[canonical]["id"]
    try:
        return gym.make(env_id)
    except Exception:
        if canonical == "LunarLander":
            return gym.make("LunarLander-v2")
        raise


__all__ = [
    "SecretKeyGameEnv",
    "TitForNTatsEnv",
    "GYM_ENVS",
    "resolve_gym_name",
    "make_gym_env",
]
