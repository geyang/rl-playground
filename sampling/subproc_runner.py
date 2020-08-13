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


class SubprocRunner:
    _msg = None

    def msg(self, msg):
        self._msg = msg

    def __init__(self, gen_fns, *args, context="spawn", context_fn=None, use_torch_mp=None, **kwargs):
        """
        :param worker:
        :param context_fn: a function you can use to create shared memory objects
        """
        if use_torch_mp is None and not USE_TORCH_MP:
            from multiprocessing import get_context
        else:
            from torch.multiprocessing import get_context

        ctx = get_context(context)
        m = ctx.Manager()
        self.manager = manager = m.__enter__()

        kw = context_fn(manager) if callable(context_fn) else {}
        kw.update(kwargs)

        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(len(gen_fns))])
        # CloundPickle(
        self.pool = [ctx.Process(target=generator_worker, args=(work_remote, remote, CloudpickleWrapper(gen), *args),
                                 kwargs=kw)
                     for work_remote, remote, gen in zip(work_remotes, self.remotes, gen_fns)]

        for p in self.pool:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for r in work_remotes:
            r.close()

    def __repr__(self):
        return f"<{self.manager} SubprocRunner>"

    def trajs(self, msg=None):
        """ yields full trajectories"""
        for r in cycle(self.remotes):
            traj = r.recv()
            r.send(msg or self._msg)
            yield traj

    def close(self):
        for p in self.pool:
            p.terminate()
            p.join()
        self.manager.__exit__(None, None, None)

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
            for step in range(limit or 1_000):
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

    runner = SubprocRunner([partial(sample_traj_gen, seed=i * 100) for i in range(1)])
    print(runner)
    with logger.time("1 env"):
        for traj in tqdm(islice(runner.trajs(), 20)):
            pass
    print('second run, should not block')
    with logger.time("1 env"):
        for traj in tqdm(islice(runner.trajs(), 20)):
            pass
    # testing the termination is important for making sure that we clean up
    del runner

    n = 10
    runner = SubprocRunner([partial(sample_traj_gen, seed=i * 100) for i in range(n)])
    print(runner)
    with logger.time(f"{n} envs"):
        for traj in tqdm(islice(runner.trajs(), 400)):
            pass

    print(traj['img'].shape)
