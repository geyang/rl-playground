import gym

from cmx import doc
from env_wrappers.flat_env import FlatEnv

from pg_experiments import instr


def inspect_rendering_on_cluster():
    """Inspect the rendering on the cluster."""
    from ml_logger import logger

    doc.new(filename="README.md", logger=logger)
    doc @ """
    # Test the Rendering from The Environment

    Currently the initial frame after reset somehow always "flashes"
    for a frame. Need to reproduce and fix.
    """
    limit = 10
    with doc, doc.row() as row:
        env_id = "ge_world:Peg2D-v0"
        env = gym.make(env_id, free=False, obs_keys=['x', 'goal'])
        for ep in range(2):
            obs, done, i = env.reset(), None, 0
            while not done and i < limit:
                act = env.action_space.sample()
                obs, reward, done, info = env.step(act)
                img = env.render("rgb", width=80, height=80)
                row.image(img)
                i += 1

    doc.flush()


def run():
    inspect_rendering_on_cluster()


if __name__ == '__main__':
    import sys
    import jaynes

    jaynes.config("local" if "pydevd" in sys.modules else "cpu-mars", launch=dict(timeout=1000))
    jaynes.run(instr(run))
