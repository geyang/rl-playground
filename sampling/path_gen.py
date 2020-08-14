import gym
import numpy as np
from collections import defaultdict

from typing import Sequence


def path_gen(env_id, seed: int, *wrappers, policy=None, obs_keys=tuple(),
             collect=None, limit=None, **env_kwargs):
    """
    :env_id: str the gym environment id
    :*wrappers: positional arguments for the wrapper
    :policy: a policy function that takes in policy(*obs[k] for k in obs_keys)
    :obs_keys: a tuple or None. When None, the obs is passed in directly
    :collect: a tuple or None. When None, defaults back to obs_keys
    :limit: the step limit, if the env does not finish before this, there is no traj['done']
    :**env_kwargs: keyvalue arguments for the environment constructor.

    :yields: {
        *collected: numpy array. np.stacked from list of values.
        'success': True  # if the episode is successful. Otherwise returns nothing.
    }
    """

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
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action = policy(*[obs[k] for k in obs_keys])
                obs, reward, done, info = env.step(action)
                # path['r'].append(- l2(obs['x'], old_obs['x']))  # * a_scale(env.spec.id))
                path['a'].append(action)  # info: add action to data set.
                path['done'].append(done)
                if not obs_keys:
                    path['x'].append(obs)
                for k in collect or []:
                    path[k].append(obs.get(k, None))
                if done:
                    path['success'] = True
                    break

            new_limit = yield {k: np.stack(v, axis=0) if isinstance(v, Sequence) else v
                               for k, v in path.items()}
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
