import copy
import os
from copy import deepcopy
import gym
import gym_maze
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange

def plot_value(q_values, losses, fig_prefix, title=None, doc=doc):
    values = q_values.max(axis=0)
    values = values.reshape((5, 5))
    plt.imshow(values)
    if title:
        plt.title(title)
    plt.colorbar()
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()

    plt.plot(losses)
    plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
    plt.title("Loss")
    plt.xlabel('Optimization Steps')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}_loss.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()

def state_to_id(state):
    x, y = state
    return 5*x + y

def id_to_state(id):
    x = id//5
    y = id % 5
    return np.array([x, y])

def get_discrete_mdp():
    env = gym.make('maze-v0')
    env.reset()
    num_states = 25
    num_actions = 4

    rewards = np.zeros((num_actions, num_states))
    dones = np.zeros((num_actions, num_states))

    dyn_mats = np.zeros((num_actions, num_states, num_states))

    for state_id in range(num_states):
        this_state = id_to_state(state_id)
        for action in range(num_actions):
            env.reset_state(copy.deepcopy(this_state))
            new_state, reward, done, _ = env.step(action)
            rewards[action, state_id] = reward
            dones[action, state_id] = float(done)
            next_state_id = state_to_id(new_state)
            dyn_mats[action, state_id, next_state_id] = 1.0

    del(env)
    return rewards, dones, dyn_mats


def perform_vi(rewards, dyn_mats, dones, gamma=0.99, eps=1e-5):
    # Assume discrete actions and states
    q_values = np.zeros(dyn_mats.shape[:2])

    deltas = []
    while not deltas or deltas[-1] >= eps:
        old = q_values
        q_max = q_values.max(axis=0)
        q_values = rewards + gamma * (1 - dones) * (dyn_mats @ q_max)

        deltas.append(np.abs(old - q_values).max())

    return q_values, deltas

class Q_table_wrapper:
    def __init__(self, q_vals):
        self.q_vals = q_vals

    def __call__(self, state):
        state = state.cpu().numpy().flatten()
        # Normalizing it back
        state = 4*state
        idx = int(state_to_id(state))
        return torch.FloatTensor(self.q_vals[:, idx])

class LFF(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = mapping_size * 2
        self.linear = nn.Linear(input_dim, self.output_dim)
        nn.init.uniform_(self.linear.weight, -scale / self.input_dim, scale / self.input_dim)
        nn.init.uniform_(self.linear.bias, -1, 1)

    def forward(self, x):
        x = self.linear(x)
        return torch.sin(2 * np.pi * x)


class RFF(LFF):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__(input_dim, mapping_size, scale=scale)
        self.linear.requires_grad = False

def perform_deep_vi(Q, rewards, dyn_mats, lr=1e-4, gamma=0.99, n_epochs=400, target_freq=10):
    Q_target = deepcopy(Q) if target_freq else Q

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    # Normalized state
    states = np.array([[x/4.0, y/4.0] for x in range(5) for y in range(5)])
    states = torch.FloatTensor(states)
    rewards = torch.FloatTensor(rewards)
    dyn_mats = torch.FloatTensor(dyn_mats)

    losses = []

    for epoch in trange(n_epochs + 1):
        if target_freq and epoch % target_freq == 0:
            Q_target.load_state_dict(Q.state_dict())

        q_max, actions = Q_target(states).max(dim=-1)
        td_target = rewards + gamma * dyn_mats @ q_max.detach()
        td_loss = l1(Q(states), td_target.T)
        losses.append(td_loss.detach().numpy())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    return q_values, losses
#
#
# def kernel_Q(q_values, states):
#     pass
#
#
def eval_q_policy(q, num_eval=100):
    """Assumes discrete action such that policy is derived by argmax a Q(s,a)"""
    torch.manual_seed(0)
    env = gym.make('maze-v0')
    returns = []

    for i in range(num_eval):
        done = False
        obs = env.reset()
        total_rew = 0
        while not done:
            obs = torch.FloatTensor(obs/4.0).unsqueeze(0)
            q_max, action = q(obs).max(dim=-1)
            obs, rew, done, _ = env.step(action.item())
            total_rew += rew
        returns.append(total_rew)

    return np.mean(returns)


if __name__ == "__main__":
    doc @ """
    ## Tabular Q-learning (Ground-truth)

    Here is the ground truth value function generated via tabular
    value iteration. It shows even for simple dynamics, the value
    function can be exponentially more complex.
    """
    from matplotlib import pyplot as plt

    with doc:
        torch.manual_seed(0)
        rewards, dones, dyn_mats = get_discrete_mdp()
        q_values, losses = perform_vi(rewards, dyn_mats, dones)

    plot_value(q_values, losses, fig_prefix="value_iteration", title="Value Iteration on Maze", doc=doc.table().figure_row())

    with doc:
        mean_reward = eval_q_policy(Q_table_wrapper(q_values))
        doc.print(f"Return with ground truth q function is {mean_reward}")

    gt_q_values = q_values  # used later

    doc @ """
    ## DQN w/ Function Approximator

    Here we plot the value function learned via deep Q Learning
    (DQN) using a neural network function approximator.
    """

    with doc:
        def get_Q_mlp():
            return nn.Sequential(
                nn.Linear(2, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 4),
            )


        Q = get_Q_mlp()
        q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, n_epochs=400)
        returns = eval_q_policy(Q)
        doc.print(f"Avg return for DQN is {returns}")

    plot_value(q_values, losses, fig_prefix="dqn", title="DQN on Maze", doc=doc.table().figure_row())

    with doc:
        Q = get_Q_mlp()
        q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, n_epochs=4000)
        returns = eval_q_policy(Q)
        doc.print(f"Avg return for DQN (4000 epochs) is {returns}")

    plot_value(q_values, losses, fig_prefix="dqn_2000", title="DQN on Maze (4000 epochs)", doc=doc.table().figure_row())

    with doc:
        def get_Q_rff(B_scale):
            return nn.Sequential(
                RFF(2, 200, scale=B_scale),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 4),
            )

    doc @ """
    ## DQN with RFF

    We can now apply this to DQN and it works right away! Using scale of 5
    """
    with doc:
        b_scale = 1
        Q = get_Q_rff(B_scale=b_scale)
        q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, n_epochs=400)
        returns = eval_q_policy(Q)

        doc.print(f"Avg return for DQN+RFF is {returns}")

    plot_value(q_values, losses, fig_prefix=f"dqn_rff_{b_scale}", title=f"DQN RFF $\sigma={b_scale}$", doc=doc.table().figure_row())
    doc.flush()
