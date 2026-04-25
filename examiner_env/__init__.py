"""
examiner_env — The Examiner RL environment package.

OpenEnv registration is deferred to after the environment class is fully
defined in environment.py.  Import this package to access KB, models,
parser, simulator, oracle, reward, baselines, and the environment itself.
"""
from importlib import import_module as _import


def _register():
    try:
        import openenv  # type: ignore
        env_mod = _import("examiner_env.environment")
        openenv.register(
            id="ExaminerEnv-v0",
            entry_point="examiner_env.environment:ExaminerEnv",
            max_episode_steps=10,
        )
    except Exception:
        pass  # registration deferred until openenv is installed


_register()
