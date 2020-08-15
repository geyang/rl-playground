import numpy as np
import torch
import torch.nn.functional as F
import gym
import time
from firedup.algos.td3 import core
from firedup.wrappers import env_fn


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""

TD3 (Twin Delayed DDPG)

"""

# def attach(fn, **kwargs):
#     for k, v in kwargs.items():
#         setattr(fn, k, v)
#     return fn


# dash-config for the dashboard
_CONFIG = dict(charts=['EpRet/mean', 'VVals/mean', 'LogPi/mean'])


def td3(env_id,
        seed=0,
        env_kwargs=dict(),
        wrappers=tuple(),
        actor_critic=core.ActorCritic, ac_kwargs=dict(),
        steps_per_epoch=5000,
        epochs=100,
        replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2,
        ep_limit=1000,
        save_freq=1,
        video_interval=None,
        _config=_CONFIG
        ):
    """

    Args:
        env_id : A gym environment id

        actor_critic: The agent's main model which for state ``x`` and
            action, ``a`` returns the following outputs:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q1``       (batch,)          | Gives one estimate of Q* for
                                           | states in ``x`` and actions in
                                           | ``a``.
            ``q2``       (batch,)          | Gives another estimate of Q* for
                                           | states in ``x`` and actions in
                                           | ``a``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                           | ``pi`` for states in ``x``:
                                           | q1(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        ep_limit (int): Maximum length of trajectory / episode / rollout.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        video_interval (int): saves the last epoch if -1, do not save if None,
            otherwise by the integer interval
    """
    from ml_logger import logger
    logger.upload_file(__file__)

    logger.save_yaml(_config, ".charts.yml")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = env_fn(env_id, *wrappers, **env_kwargs, seed=seed)
    test_env = env_fn(env_id, *wrappers, **env_kwargs, seed=seed + 100)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Main outputs from computation graph
    main = actor_critic(in_features=obs_dim, **ac_kwargs)

    # Target policy network
    target = actor_critic(in_features=obs_dim, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in
                       [main.policy, main.q1, main.q2, main])
    logger.print("Number of parameters: \t pi: {:d}, \t q1: {:d}, \t q2: {:d}, \t total: {:d}\n".format(*var_counts))

    # Separate train ops for pi, q
    pi_optimizer = torch.optim.Adam(main.policy.parameters(), lr=pi_lr)

    q_params = list(main.q1.parameters()) + list(main.q2.parameters())
    q_optimizer = torch.optim.Adam(q_params, lr=q_lr)

    # Initializing targets to match main variables
    target.load_state_dict(main.state_dict())

    def get_action(o, noise_scale):
        pi = main.policy(torch.Tensor(o.reshape(1, -1)))
        a = pi.detach().numpy()[0] + noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10, log_video=False, epoch=None):
        frames = []
        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == ep_limit)):
                # Take deterministic actions at test time
                o, r, d, info = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
                if log_video:
                    frames.append(test_env.render("rgb_array"))
            logger.store(EpRet=ep_ret, EpLen=ep_len, **info, prefix="test/")
            if log_video:
                logger.save_video(frames, f"videos/test_{epoch}.mp4")

    logger.start('start', 'epoch')
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps + 1):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy (with some noise, via act_noise).
        """
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == ep_limit else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == ep_limit):
            """
            Perform all TD3 updates at the end of the trajectory
            (in accordance with source code of TD3 published by
            original authors).
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                obs1, obs2, acts, rews, done = (torch.tensor(batch['obs1']),
                                                torch.tensor(batch['obs2']),
                                                torch.tensor(batch['acts']),
                                                torch.tensor(batch['rews']),
                                                torch.tensor(batch['done']))
                q1 = main.q1(torch.cat((obs1, acts), dim=1))
                q2 = main.q2(torch.cat((obs1, acts), dim=1))
                pi_targ = target.policy(obs2)

                # Target policy smoothing, by adding clipped noise to target actions
                epsilon = torch.normal(torch.zeros_like(pi_targ),
                                       target_noise * torch.ones_like(pi_targ))

                epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
                a2 = torch.clamp(pi_targ + epsilon, -act_limit, act_limit)

                # Target Q-values, using action from target policy
                q1_targ = target.q1(torch.cat((obs2, a2), dim=1))
                q2_targ = target.q2(torch.cat((obs2, a2), dim=1))

                # Bellman backup for Q functions, using Clipped Double-Q targets
                min_q_targ = torch.min(q1_targ, q2_targ)
                backup = (rews + gamma * (1 - done) * min_q_targ).detach()

                # TD3 Q losses
                q1_loss = F.mse_loss(q1, backup)
                q2_loss = F.mse_loss(q2, backup)
                q_loss = q1_loss + q2_loss

                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                logger.store(LossQ=q_loss.item(), Q1Vals=q1.detach().numpy(),
                             Q2Vals=q2.detach().numpy())

                if j % policy_delay == 0:
                    q1_pi = main.q1(torch.cat((obs1, main.policy(obs1)), dim=1))

                    # TD3 policy loss
                    pi_loss = -q1_pi.mean()

                    # Delayed policy update
                    pi_optimizer.zero_grad()
                    pi_loss.backward()
                    pi_optimizer.step()

                    # Polyak averaging for target variables
                    for p_main, p_target in zip(main.parameters(), target.parameters()):
                        p_target.data.copy_(polyak * p_target.data + (1 - polyak) * p_main.data)

                    logger.store(LossPi=pi_loss.item())

            logger.store(EpRet=ep_ret, EpLen=ep_len, **info)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs - 1):
            #     logger.save_state({'env': env}, main, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(log_video=video_interval and epoch % video_interval == 0, epoch=epoch)

            # Log info about epoch
            logger.log_metrics_summary(
                key_values={'epoch': epoch, 'envSteps': t, 'time': logger.since('start')},
                key_stats={'EpRet': "min_max", 'TestEpRet': "min_max", 'EpLen': "mean",
                           'TestEpLen': "mean", 'Q1Vals': "min_max", 'Q2Vals': "min_max",
                           'LossPi': "mean", 'LossQ': "mean", })


if __name__ == '__main__':
    td3("HalfCheetah-v2", actor_critic=core.ActorCritic,
        ac_kwargs=dict(hidden_sizes=[300] * 2), gamma=0.99, seed=0, epochs=50)
