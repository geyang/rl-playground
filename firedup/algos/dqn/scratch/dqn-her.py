from collections import deque

import numpy as np
import gym
import torch
import torch.nn.functional as F
from firedup.algos.dqn import core
from firedup.wrappers import env_fn
from itertools import chain
from more_itertools import ichunked


class HerReplayBuffer:
    """
    Does online high-sight experience replay

    issues: should replace randomly as opposed to FIFO
    """

    def __init__(self, obs_dim, act_dim, size):
        self.data = deque(maxlen=size)

    def append(self, traj):
        self.data.append(traj)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        traj_inds = np.random.randint(0, len(self))
        rand_trajs = (self.data[i] for i in traj_inds)
        # todo: HER logic goes here
        yield from rand_trajs

    # usage: sample("x", "x@1:", "r", "done")
    def sample(self, *keys, batch_size, n_step=1):
        """currently does traj-level shuffling. n_step is for n-step Q-learning"""
        next = slice(1, )
        for traj in self:
            yield from zip(eval("traj[{}][{}]".format(*k.split("@")))
                           if "@" in k else traj[k] for k in keys)


"""
Deep Q-Network + HER + Pixel input
"""
def dqn(env_id,
        seed=0,
        num_envs=8,
        obs_keys=None,
        q_network=core.QMlp, ac_kwargs={},
        steps_per_epoch=5000, epochs=100,
        replay_size=int(1e6), gamma=0.99,
        min_replay_history=20000,
        epsilon_decay_period=250000, epsilon_train=0.01,
        epsilon_eval=0.001, lr=1e-3, ep_limit=1000,
        update_interval=4, target_update_interval=8000,
        batch_size=100, save_freq=1, ):
    from ml_logger import logger
    logger.log_params(kwargs=locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.log_text("""
                    charts:
                    - xKey: __timestamp
                      xFormat: time
                      yKey: EpRet/mean
                    - xKey: epoch
                      yKey: LossQ/mean
                    """, ".charts.yml", overwrite=True)

    env, test_env = env_fn(env_id, seed=seed), env_fn(env_id, seed=seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = 1  # env.action_space.shape

    # Share information about action space with policy architecture
    # ac_kwargs["action_space"] = env.action_space

    # Main computation graph
    main = q_network(in_features=obs_dim, **ac_kwargs)

    # Target network
    target = q_network(in_features=obs_dim, **ac_kwargs)

    # Experience buffer
    buffer = HerReplayBuffer(size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [main.q, main])
    logger.print("Number of parameters: \t q: {:d}, \t total: {:d}\n".format(*var_counts), color="green")

    # Value train op
    value_params = main.q.parameters()
    value_optimizer = torch.optim.Adam(value_params, lr=lr)

    # Initializing targets to match main variables
    target.load_state_dict(main.state_dict())

    def get_action(o_s, o_g, epsilon):
        if np.random.random() <= epsilon:
            return env.action_space.sample()
        else:
            q_values = main(o_s, o_g)
            return torch.argmax(q_values, dim=1).item()

    def test_agent(n=10):
        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == ep_limit)):
                # epsilon_eval used when evaluating the agent
                o, r, d, _ = test_env.step(get_action(o, epsilon_eval))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    buffer = HerReplayBuffer()
    from sampling.subproc_runner import SubprocRunner
    from sampling.path_gen import path_gen
    from functools import partial

    env_step_limit = steps_per_epoch * epochs

    # pyTorch memory share
    policy.shape_memory()
    # todo: add shared step counter object
    sampler = SubprocRunner([partial(path_gen, env_id, seed + 100 * rank) for rank in range(num_envs)],
                            obs_keys=obs_keys, collect=obs_keys)
    # todo: eval sampler

    env_steps = 0

    # Main loop: collect experience in env and update/log each epoch
    logger.start('start', 'epoch')
    for traj in sampler.trajs():
        # for k, v in traj.items():
        #     print(f"{k}: Size{v.shape}")
        buffer.append(traj)
        steps = len(traj['a'])
        env_steps += steps

        for obs1, obs2, acts, rews, done in buffer.sample("x", "x@next", "a", "r", "done", batch_size=32):
            q_pi = main(obs1).gather(1, acts.long()).squeeze()
            q_pi_targ, _ = target(obs2).max(1)

            # Bellman backup for Q function
            backup = (rews + gamma * (1 - done) * q_pi_targ).detach()

            # DQN loss
            value_loss = F.smooth_l1_loss(q_pi, backup)

            # Q-learning update
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            logger.store(LossQ=value_loss.item(), QVals=q_pi.data.numpy())

        if env_steps >= env_step_limit:
            break

    for t in range(env_steps):
        main.eval()

        # the epsilon value used for exploration during training
        epsilon = core.linearly_decaying_epsilon(
            epsilon_decay_period, t, min_replay_history, epsilon_train
        )
        a = get_action(obs[state_key], obs[goal_key], epsilon)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == ep_limit else d

        # Store experience to replay buffer
        # replay_buffer.store(obs=obs, a=a, r=r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        obs = o2

        if d or (ep_len == ep_limit):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # train at the rate of update_interval if enough training steps have been run
        if replay_buffer.size > min_replay_history and t % update_interval == 0:
            main.train()
            batch = replay_buffer.sample_batch(batch_size)
            obs1, obs2, acts, rews, done =
            torch.Tensor(batch["obs1"]),
            torch.Tensor(batch["obs2"]),
            torch.Tensor(batch["acts"]),
            torch.Tensor(batch["rews"]),
            torch.Tensor(batch["done"])

        obs1, obs2, acts, rews, done
        q_pi = main(obs1).gather(1, acts.long()).squeeze()
        q_pi_targ, _ = target(obs2).max(1)

        # Bellman backup for Q function
        backup = (rews + gamma * (1 - done) * q_pi_targ).detach()

        # DQN loss
        value_loss = F.smooth_l1_loss(q_pi, backup)

        # Q-learning update
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        logger.store(LossQ=value_loss.item(), QVals=q_pi.data.numpy())

    # syncs weights from online to target network
    if t % target_update_interval == 0:
        target.load_state_dict(main.state_dict())

    # End of epoch wrap-up
    if replay_buffer.size > min_replay_history and t % steps_per_epoch == 0:
        epoch = t // steps_per_epoch

        # Save model
        # if (epoch % save_freq == 0) or (epoch == epochs - 1):
        #     logger.save_state({"env": env}, main, None)

        # Test the performance of the deterministic version of the agent.
        test_agent()

        # Log info about epoch
        logger.log_metrics_summary(key_values={"epoch": epoch, "envSteps": t, "time": logger.since('start')},
                                   key_stats={"EpRet": "min_max", "TestEpRet": "min_max", "EpLen": "mean",
                                              "TestEpLen": "mean", "QVals": "min_max", "LossQ": "mean"})


if __name__ == '__main__':
    # dqn("ge_world:CMaze-v0",
    dqn("LunarLander-v2",
        ac_kwargs=dict(hidden_sizes=[64, ] * 2),
        gamma=0.99,
        seed=0,
        steps_per_epoch=4000,
        epochs=50)
