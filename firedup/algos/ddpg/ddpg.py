import numpy as np
import torch
import torch.nn.functional as F
from firedup.algos.ddpg import core
from firedup.wrappers import env_fn


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
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
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )


"""

Deep Deterministic Policy Gradient (DDPG)

"""

_CONFIG = dict(charts=["EpRet/mean", "TestEpRet/mean", "QVals/mean", "LossQ/mean", "LossPi/mean", ])


def ddpg(
        env_id,
        seed=0,
        env_kwargs=dict(),
        test_env_kwargs=None,
        env_fn=env_fn,
        wrappers=tuple(),
        actor_critic=core.ActorCritic,
        ac_kwargs=dict(),
        steps_per_epoch=5000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=100,
        start_steps=10000,
        act_noise=0.1,
        ep_limit=1000,
        save_freq=1,
        video_interval=None,
        _config=_CONFIG
):
    """

    Args:
        env_id : A gym environment id

        actor_critic: The agent's main model which takes some states ``x`` and 
            and actions ``a`` and returns a tuple of:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for
                                           | states ``x`` and actions in
                                           | ``a``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and
                                           | ``pi`` for states in ``x``:
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            class you provided to DDPG.

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

    test_env_kwargs = test_env_kwargs or env_kwargs
    env = env_fn(env_id, *wrappers, **env_kwargs, seed=seed)
    test_env = env_fn(env_id, *wrappers, **test_env_kwargs, seed=seed + 100)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs["action_space"] = env.action_space

    # Main outputs from computation graph
    main = actor_critic(in_features=obs_dim, **ac_kwargs)

    # Target networks
    target = actor_critic(in_features=obs_dim, **ac_kwargs)
    target.eval()

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(
        core.count_vars(module) for module in [main.policy, main.q, main]
    )
    logger.print("Number of parameters: \t q: {:d}, \t total: {:d}\n".format(*var_counts))

    # Separate train ops for pi, q
    pi_optimizer = torch.optim.Adam(main.policy.parameters(), lr=pi_lr)
    q_optimizer = torch.optim.Adam(main.q.parameters(), lr=q_lr)

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
                logger.save_video(frames, f"videos/test_{epoch:04d}.mp4")

    logger.start("start")
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps + 1):
        main.eval()
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
            main.train()
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                obs1, obs2, acts, rews, done = (
                    torch.Tensor(batch["obs1"]),
                    torch.Tensor(batch["obs2"]),
                    torch.Tensor(batch["acts"]),
                    torch.Tensor(batch["rews"]),
                    torch.Tensor(batch["done"]),
                )
                _, q, q_pi = main(obs1, acts)
                _, q_pi_targ = target(obs2)

                # Bellman backup for Q function
                backup = (rews + gamma * (1 - done) * q_pi_targ).detach()

                # DDPG losses
                pi_loss = -q_pi.mean()
                q_loss = F.mse_loss(q, backup)

                # Policy update
                pi_optimizer.zero_grad()
                pi_loss.backward()
                pi_optimizer.step()
                logger.store(LossPi=pi_loss.item())

                # Q-learning update (note: after Policy update to avoid error)
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()
                logger.store(LossQ=q_loss.item(), QVals=q.data.numpy())

                # Polyak averaging for target parameters
                for p_main, p_target in zip(main.parameters(), target.parameters()):
                    p_target.data.copy_(
                        polyak * p_target.data + (1 - polyak) * p_main.data
                    )

            logger.store(EpRet=ep_ret, EpLen=ep_len, **info)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs - 1):
            #     logger.save_state({"env": env}, main, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(log_video=video_interval and epoch % video_interval == 0, epoch=epoch)

            # Log info about epoch
            logger.log_metrics_summary(
                key_values={"epoch": epoch, "envSteps": t, "time": logger.since("start")},
                key_stats={"EpRet": "min_max", "TestEpRet": "min_max", "EpLen": "mean",
                           "TestEpLen": "mean", "QVals": "min_max", "LossPi": "mean",
                           "LossQ": "mean", })


if __name__ == "__main__":
    ddpg("HalfCheetah-v2",
         actor_critic=core.ActorCritic,
         ac_kwargs=dict(hidden_sizes=[31, 31] * 4),
         gamma=0.99,
         seed=0,
         epochs=50,
         )
