import numpy as np
import gym
import torch
import torch.nn.functional as F
from collections import defaultdict
from firedup.algos.dqn import core
from firedup.wrappers import env_fn

"""
An Enhanced Version of dqn, with better better coding style.
"""


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
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
Deep Q-Network
"""


def dqn(env, test_env, exp_name, q_network=core.QMlp, ac_kwargs={}, seed=0, steps_per_epoch=5000, epochs=100,
        replay_size=int(1e6), gamma=0.99, min_replay_history=20000, epsilon_start=0.9,
        epsilon_end=0.01, epsilon_decay=200, epsilon_eval=0.001, lr=1e-3, momentum=0.9, ep_limit=1000, update_interval=4, target_update_interval=8000,
        batch_size=100, save_freq=1, device='cuda', save_dir=None):
    __d = locals()
    from ml_logger import ML_Logger
    logger = ML_Logger(prefix=f"rl_transfer/rl_playground/dqn/{exp_name}")
    logger.log_params(kwargs=__d)
    logger.upload_file(__file__)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)

    logger.log_text("""
                    charts:
                    - xKey: envSteps
                      yKey: EpRet/mean
                    - xKey: epoch
                      yKey: TestEpRet/mean                      
                    - xKey: epoch
                      yKey: LossQ/mean
                    """, ".charts.yml", overwrite=True)

    obs_dim = env.observation_space.shape[0]
    act_dim = 1

    # Share information about action space with policy architecture
    ac_kwargs["action_space"] = env.action_space

    # Main computation graph
    main = q_network(in_features=obs_dim, **ac_kwargs)
    main.to(device)

    # Target network
    target = q_network(in_features=obs_dim, **ac_kwargs)
    target.to(device)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [main.q, main])
    logger.print("Number of parameters: \t q: {:d}, \t total: {:d}\n".format(*var_counts), color="green")

    # Value train op
    value_params = main.q.parameters()
    value_optimizer = torch.optim.SGD(value_params, lr=lr, momentum=momentum)

    # Initializing targets to match main variables
    target.load_state_dict(main.state_dict())

    def get_action(o, epsilon):
        """Select an action from the set of available actions.
        Chooses an action randomly with probability epsilon otherwise
        act greedily according to the current Q-value estimates.
        """
        if np.random.random() <= epsilon:
            return env.action_space.sample()
        else:
            q_values = main(torch.Tensor(o.reshape(1, -1)).to(device))
            # return the action with highest Q-value for this observation
            return torch.argmax(q_values, dim=1).item()

    def test_agent(n=10):
        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == ep_limit)):
                # epsilon_eval used when evaluating the agent
                o, r, d, _ = test_env.step(get_action(o, epsilon_eval))
                ep_ret += r
                ep_len += 1
            logger.store(TestMeanRew=ep_ret/ep_len, TestEpRet=ep_ret, TestEpLen=ep_len)

    total_steps = steps_per_epoch * epochs

    logger.start('start', 'epoch')
    # this is an online version
    # Main loop: collect experience in env and update/log each epoch
    num_episodes=0
    done, traj = True, None
    for t in range(total_steps + 1):
        if done or traj['a'].__len__() == ep_limit:
            num_episodes += 1
            if traj:
                logger.store(MeanRew=np.mean(traj['r']), EpRet=sum(traj['r']), EpLen=len(traj['a']))
            obs = env.reset()
            traj = defaultdict(list, {"x": [obs]})

        main.eval()

        # the epsilon value used for exploration during training
        epsilon = core.exponential_decaying_epsilon(
            epsilon_start, epsilon_end, epsilon_decay, num_episodes, min_replay_history,
        )
        a = get_action(obs, epsilon)
        traj['a'].append(a)

        # Step the env
        obs, r, done, _ = env.step(a)
        done = False if len(traj['a']) == ep_limit else done
        traj['x'].append(obs)
        traj['r'].append(r)
        traj['done'].append(done)

        # todo: insert online HER relabeling here
        replay_buffer.store(traj['x'][-1], a, r, obs, done)

        # train at the rate of update_interval if enough training steps have been run
        if replay_buffer.size > min_replay_history and t % update_interval == 0:
            main.train()
            batch = replay_buffer.sample_batch(batch_size)
            (obs1, obs2, acts, rews, dones) = (
                torch.Tensor(batch["obs1"]).to(device),
                torch.Tensor(batch["obs2"]).to(device),
                torch.Tensor(batch["acts"]).to(device),
                torch.Tensor(batch["rews"]).to(device),
                torch.Tensor(batch["done"]).to(device),
            )
            q_pi = main(obs1).gather(1, acts.long()).squeeze()
            q_pi_targ, _ = target(obs2).max(1)

            # Bellman backup for Q function
            backup = (rews + gamma * (1 - dones) * q_pi_targ).detach()

            # DQN loss
            value_loss = F.smooth_l1_loss(q_pi, backup)

            # Q-learning update
            value_optimizer.zero_grad()
            value_loss.backward()

            # Clamp gradients
            for param in value_params:
                param.grad.data.clamp_(-1, 1)

            value_optimizer.step()
            logger.store(LossQ=value_loss.item(), QVals=q_pi.data.cpu().numpy())

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
                                       key_stats={"EpRet": "min_max", "MeanRew":"min_max", "TestMeanRew":"min_max", "TestEpRet": "min_max", "EpLen": "mean",
                                                  "TestEpLen": "mean", "QVals": "min_max", "LossQ": "mean"})
        state_dict = {'q_net':main.state_dict()}
        torch.save(state_dict, f'{save_dir}/state.pt')
