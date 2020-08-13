import gym
import numpy as np
from collections import defaultdict


def path_gen(env_id, seed: int, *wrappers, policy=None, obs_keys=tuple(),
             collect=None, limit=None, **env_kwargs):
    # todo: add whildcard `*` for obs_keys
    env = gym.make(env_id, **env_kwargs)
    for w in wrappers:
        env = w(env)
    env.seed(seed)

    collect = collect or obs_keys
    try:
        while True:
            obs, done = env.reset(), False
            d = {k: [obs[k]] for k in collect if k in obs} if collect else {"x": [obs]}
            path = defaultdict(list, d)
            for step in range(limit or 10):
                # uniform sampler
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action = policy.act(*[obs[k] for k in obs_keys])
                obs, reward, done, info = env.step(action)
                # path['r'].append(- l2(obs['x'], old_obs['x']))  # * a_scale(env.spec.id))
                path['a'].append(action)  # info: add action to data set.
                if not obs_keys:
                    path['x'].append(obs)
                for k in collect or []:
                    path[k].append(obs.get(k, None))
                if done:
                    break

            new_limit = yield {k: np.stack(v, axis=0) for k, v in path.items()}
            if new_limit is not None:
                limit = new_limit


    finally:
        print('clean up the environment')
        env.close()


if __name__ == '__main__':
    gen = path_gen("Reacher-v2", seed=100)
    while True:
        traj = next(gen)
        for k, v in traj.items():
            print(f"{k}: Size{v.shape}")
        traj = gen.send(10)
        for k, v in traj.items():
            print(f"{k}: Size{v.shape}")
        break
    # for traj in gen:
    #     for k, v in traj.items():
    #         print(f"{k}: Size{v.shape}")
    #     break
