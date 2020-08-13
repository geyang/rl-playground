from itertools import cycle

"""An RL Sampling Running built with (pyTorch) Multiprocessing"""

USE_TORCH_MP = True


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, payload):
        self.payload = payload

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.payload)

    def __setstate__(self, ob):
        import pickle
        self.payload = pickle.loads(ob)


def generator_worker(remote, parent_remote, wrapped: CloudpickleWrapper, *args, **kwargs):
    parent_remote.close()
    traj_gen = wrapped.payload
    gen = traj_gen(*args, **kwargs)
    traj = next(gen)
    while True:
        remote.send(traj)
        msg = remote.recv()
        traj = gen.send(msg)


class DummyRunner:
    _msg = None

    def msg(self, msg):
        self._msg = msg

    def __init__(self, gen_fns, *args, context_fn=None, **kwargs):
        """
        :param worker:
        :param context_fn: a function you can use to create shared memory objects
        """
        kw = context_fn(None) if callable(context_fn) else {}
        kw.update(kwargs)
        self.gen_fns = [f(*args, **kw) for f in gen_fns]

    def __repr__(self):
        return f"<DummyRunner>"

    def trajs(self, msg=None):
        """ yields full trajectories"""
        # if limit is not None:
        #     self._msg = limit
        for gen in cycle(self.gen_fns):
            yield next(gen) if msg is None else gen.send(msg)

    def close(self):
        for gen in self.gen_fns:
            gen.close()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    import gym
    import numpy as np
    from tqdm import tqdm
    from functools import partial
    from itertools import islice
    from collections import defaultdict


    def sample_traj_gen(seed, limit=None, **context_args):
        env = gym.make("Reacher-v2")
        env.seed(seed)

        while True:
            obs = env.reset()
            traj = defaultdict(list, obs=[obs])
            act, done = env.action_space.sample(), False
            for step in range(limit or 1000):
                obs, r, done, info = env.step(act)
                traj['obs'].append(obs)
                traj['r'].append(r)
                # traj['info'].append(info)
                img = env.render("rgb_array", width=84, height=84)
                traj['img'].append(img)
                if done:
                    break

            new_limit = yield {k: np.stack(v, axis=0) for k, v in traj.items()}
            if new_limit is not None:
                limit = new_limit


    from ml_logger import logger

    runner = DummyRunner([partial(sample_traj_gen, seed=i * 100, limit=4) for i in range(1)])
    print(runner)
    with logger.time("1 env"):
        for traj in tqdm(islice(runner.trajs(), 10)):
            pass
    with logger.time("1 env"):
        for traj in tqdm(islice(runner.trajs(10), 10)):
            pass
    # testing the termination is important for making sure that we clean up
    del runner
