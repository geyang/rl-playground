from collections import defaultdict, deque

import numpy as np
import gym
import torch
import torch.nn.functional as F
from firedup.algos.dqn import core
from firedup.wrappers import env_fn


class SimpleBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
    """

    def __init__(self, size):
        self.data = deque(maxlen=size)

    def store(self, *args):
        self.data.append(args)

    def __len__(self):
        return self.data.__len__()

    def sample(self, batch_size=32):
        """with replacement."""
        inds = np.random.randint(0, len(self), size=batch_size)
        results = []
        for arg in zip(*[self.data[i] for i in inds]):
            first = arg[0]
            if isinstance(first, dict):
                results.append({k: np.stack([a[k] for a in arg]) for k in first})
            else:
                results.append(np.stack(arg))
        return results


"""
Deep Q-Network
"""


def dqn(env_id,
        env_kwargs={},
        obs_keys=None,  # we support slice, str etc.
        her_k=None,
        optim_epochs=1,
        q_network=core.QMlp, ac_kwargs={}, seed=0, steps_per_epoch=5000, epochs=100,
        replay_size=int(1e6), gamma=0.99, min_replay_history=20000,
        epsilon_decay_period=250000, epsilon_train=0.01,
        epsilon_eval=0.001,
        lr=1e-3, batch_size=100, update_interval=4,
        max_ep_len=1000, target_update_interval=8000,
        save_freq=1, ):
    __d = locals()
    from ml_logger import logger
    logger.log_params(kwargs=__d)
    logger.upload_file(__file__)

    assert min_replay_history < replay_size, "min replay history need to be smaller than the buffer size"

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.log_text("""
                    charts:
                    - xKey: __timestamp
                      xFormat: time
                      yKey: EpRet/mean
                    - xKey: __timestamp
                      xFormat: time
                      yKey: test/EpRet/mean
                    - xKey: __timestamp
                      xFormat: time
                      yKey: test/success/mean
                    - xKey: epoch
                      yKey: LossQ/mean
                    """, ".charts.yml", dedent=True, overwrite=True)

    env = env_fn(env_id, seed=seed, **env_kwargs)
    test_env = env_fn(env_id, seed=seed + 100, **env_kwargs)

    if obs_keys:
        obs_dim = sum([env.observation_space[k].shape[0] for k in obs_keys])
    else:
        obs_dim = env.observation_space.shape[0]
    # Share information about action space with policy architecture
    ac_kwargs["action_space"] = env.action_space

    # Main computation graph
    main = q_network(in_features=obs_dim, **ac_kwargs)

    # Target network
    target = q_network(in_features=obs_dim, **ac_kwargs)

    # Experience buffer
    replay_buffer = SimpleBuffer(size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [main.q, main])
    logger.print("Number of parameters: \t q: {:d}, \t total: {:d}\n".format(*var_counts), color="green")

    # Value train op
    value_params = main.q.parameters()
    value_optimizer = torch.optim.Adam(value_params, lr=lr)

    # Initializing targets to match main variables
    target.load_state_dict(main.state_dict())

    def get_action(*obs, eps):
        """Select an action from the set of available actions.
        Chooses an action randomly with probability epsilon otherwise
        act greedily according to the current Q-value estimates.
        """
        if eps and np.random.random() <= eps:
            return env.action_space.sample()
        else:
            # returns [9] size tensor
            q_values = main(*[torch.Tensor(o) for o in obs])
            # return the action with highest Q-value for this observation
            return torch.argmax(q_values, dim=-1).item()

    def test_agent(n=10):
        for _ in range(n):
            obs, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # epsilon_eval used when evaluating the agent
                act = get_action(*unpack(obs, obs_keys), eps=epsilon_eval)
                obs, r, d, _ = test_env.step(act)
                ep_ret += r
                ep_len += 1
            success = False if ep_len == max_ep_len else d
            logger.store(EpRet=ep_ret, EpLen=ep_len, success=success, prefix="test/")

    total_steps = steps_per_epoch * epochs

    def unpack(obs, obs_keys=None):
        return obs if obs_keys is None else [obs[k] for k in obs_keys]

    logger.start('start', 'epoch')
    # this is an online version
    # Main loop: collect experience in env and update/log each epoch
    done, traj = True, None
    for t in range(total_steps):
        if done or traj['a'].__len__() == max_ep_len:
            if traj:
                logger.store(EpRet=sum(traj['r']), EpLen=len(traj['a']), success=max(traj['done']))
            obs = env.reset()
            traj = defaultdict(list, {"x": [obs]})

        main.eval()

        # the epsilon value used for exploration during training
        epsilon = core.linearly_decaying_epsilon(
            epsilon_decay_period, t, min_replay_history, epsilon_train
        )
        a = get_action(*unpack(obs, obs_keys), eps=epsilon)
        traj['a'].append(a)

        # Step the env
        obs, r, done, _ = env.step(a)
        done = False if traj['a'].__len__() == max_ep_len else done
        traj['x'].append(obs)
        traj['r'].append(r)
        traj['done'].append(done)

        # done: insert online HER relabeling here
        replay_buffer.store(traj['x'][-1], a, r, obs, done)
        if her_k:
            _ = traj['x'][-2::-her_k], traj['x'][-1::-her_k]
            for step, (obs1, obs2) in enumerate(zip(*_)):
                _obs1, _obs2 = obs1.copy(), obs2.copy()
                _obs1['goal'] = _obs2['goal'] = obs['x']
                replay_buffer.store(_obs1, a, -1 if step else 0, _obs2, False if step else True)
                if step:  # stop the relabel after 1-step
                    break

        # train at the rate of update_interval if enough training steps have been run
        if len(replay_buffer) > min_replay_history and t % update_interval == 0:
            main.train()
            for optim_step in range(optim_epochs):
                batch = replay_buffer.sample(batch_size)
                obs1, acts, rews, obs2, dones = [
                    {k: torch.Tensor(v) for k, v in a.items()} if isinstance(a, dict) else torch.Tensor(a)
                    for a in batch]
                q_pi = main(*unpack(obs1, obs_keys)).gather(-1, acts[:, None].long()).squeeze()
                q_pi_targ, _ = target(*unpack(obs2, obs_keys)).max(1)

                # Bellman backup for Q function
                backup = (rews + gamma * (1 - dones) * q_pi_targ).detach()

                # DQN loss
                value_loss = F.smooth_l1_loss(q_pi, backup)

                # Q-learning update
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
                logger.store(LossQ=value_loss.item(), QVals=q_pi.data.numpy(), eps=epsilon)

        # syncs weights from online to target network
        if t % target_update_interval == 0:
            target.load_state_dict(main.state_dict())

        # End of epoch wrap-up
        if len(replay_buffer) > min_replay_history and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs - 1):
            #     logger.save_state({"env": env}, main, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_metrics_summary(key_values={"epoch": epoch, "envSteps": t, "time": logger.since('start')},
                                       key_stats={"EpRet": "min_max", "EpLen": "mean", "success": "mean",
                                                  "test/EpRet": "min_max", "test/EpLen": "mean", "test/success": "mean",
                                                  "eps": "mean", "QVals": "min_max", "LossQ": "mean"})


if __name__ == '__main__':
    dqn("ge_world:CMaze-discrete-v0",
        obs_keys=["x", "goal"],
        ac_kwargs=dict(hidden_sizes=[64, ] * 2),
        gamma=0.99,
        seed=0,
        steps_per_epoch=4000,
        epochs=50)
