import gym
from copy import deepcopy, copy


def env_fn(env_id, *wrappers, seed=None, **kwargs):
    env = gym.make(env_id, **kwargs)
    if seed:
        env.seed(seed)
    for w in wrappers:
        env = w(env)
    return env


env_cache = {}


def singleton_env_fn(env_id, *wrappers, seed=None, **kwargs):
    """Used with metaworld fixed environments to return
    the same environment instance. Environments are keyed
    by the environment id."""
    if env_id in env_cache:
        return deepcopy(env_cache[env_id])

    env = gym.make(env_id, **kwargs)
    if seed:
        env.seed(seed)
    for w in wrappers:
        env = w(env)
    env_cache[env_id] = env
    return env


if __name__ == '__main__':
    from cmx import doc

    with doc, doc.row() as row:
        env = singleton_env_fn("env_wrappers.metaworld:Reach-fixed-v1", seed=100)
        env.reset()
        for i in range(2):
            a = env.action_space.sample()
            env.step(a)
            img = env.render('rgb', width=240, height=240)
        row.image(img, caption="Env 1")

        env = singleton_env_fn("env_wrappers.metaworld:Reach-fixed-v1")
        env.reset()
        a = env.action_space.sample()
        env.step(a)
        img = env.render('rgb', width=240, height=240)
        row.image(img, caption="Env 2")
