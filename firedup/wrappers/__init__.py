import gym


def env_fn(env_id, *wrappers, seed=None, **kwargs):
    env = gym.make(env_id, **kwargs)
    if seed:
        env.seed(seed)
    for w in wrappers:
        env = w(env)
    return env
