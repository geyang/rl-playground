import copy
import os
from copy import deepcopy
import gym
import gym_maze
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange

class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class View(nn.Module):
    def __init__(self, *dims, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dims = dims

    def forward(self, x):
        if self.batch_first:
            return x.view(-1, *self.dims)
        else:
            return x.view(*self.dims)

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

class LFF(nn.Module):
    def __init__(self, input_feats, output_feats, kernel_dim,  stride, scale=1.0):
        super().__init__()
        self.in_features = input_feats
        self.out_features = output_feats
        self.conv = nn.Conv2d(input_feats, output_feats, kernel_dim, stride=stride)
        nn.init.uniform_(self.conv.weight, -scale / (input_feats*kernel_dim*kernel_dim), scale / (input_feats*kernel_dim*kernel_dim))
        nn.init.uniform_(self.conv.bias, -1, 1)

    def forward(self, x):
        x = self.conv(np.pi * x)
        return torch.sin(x)


class RFF(LFF):
    def __init__(self, input_feats, output_feats, kernel_dim,  stride, scale=1.0):
        super().__init__(input_feats, output_feats, kernel_dim,  stride, scale)
        self.conv.requires_grad = False

def perform_deep_vi(Q, rewards, dyn_mats, dones, lr=1e-4, gamma=0.99, n_epochs=400, target_freq=50):
    Q_target = deepcopy(Q) if target_freq else Q
    env = gym.make('maze-v0')
    env.reset()
    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = []
    for x in range(5):
        for y in range(5):
            env.reset_state(np.array([x, y]))
            state = env.render()
            state = cv2.resize(state, dsize=(64, 64), interpolation=cv2.INTER_AREA)
            state = np.transpose(state, (2, 0, 1))
            states.append(state)

    del(env)
    states = np.array(states)
    states = torch.FloatTensor(states)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)
    dyn_mats = torch.FloatTensor(dyn_mats)

    losses = []

    for epoch in trange(n_epochs + 1):
        if target_freq and epoch % target_freq == 0:
            Q_target.load_state_dict(Q.state_dict())

        q_max, actions = Q_target(states).max(dim=1)
        td_target = rewards + gamma * (1-dones) * (dyn_mats @ q_max.detach())
        td_loss = l1(Q(states), td_target.T)
        losses.append(td_loss.detach().numpy())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    return q_values, losses

def eval_q_policy(q, num_eval=100):
    """Assumes discrete action such that policy is derived by argmax a Q(s,a)"""
    torch.manual_seed(0)
    env = gym.make('maze-v0')
    returns = []

    for i in range(num_eval):
        done = False
        env.reset()
        obs = env.render()
        obs = cv2.resize(obs, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        obs = np.transpose(obs, (2, 0, 1))
        total_rew = 0
        while not done:
            obs = torch.FloatTensor(obs).unsqueeze(0)
            q_max, action = q(obs).max(dim=-1)
            obs, rew, done, _ = env.step(action.item())
            obs = env.render()
            obs = cv2.resize(obs, dsize=(64, 64), interpolation=cv2.INTER_AREA)
            obs = np.transpose(obs, (2, 0, 1))
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

    doc @ """
    ## DQN w/ Function Approximator

    Here we plot the value function learned via deep Q Learning
    (DQN) using a neural network function approximator.
    """

    with doc:
        def get_Q_mlp():
            return nn.Sequential(
                Lambda(lambda x: x / 255),
                # RFF(3, 8, 3, stride=1),
                nn.Conv2d(3, 8, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, stride=1),
                nn.ReLU(),
                View(8*58*58),
                nn.Linear(8*58*58, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 4),
            )


        Q = get_Q_mlp()
        q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, dones, n_epochs=1500)
        returns = eval_q_policy(Q)
        doc.print(f"Avg return for DQN is {returns}")

    plot_value(q_values, losses, fig_prefix="dqn", title="DQN on Maze", doc=doc.table().figure_row())

    # with doc:
    #     Q = get_Q_mlp()
    #     q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, n_epochs=4000)
    #     returns = eval_q_policy(Q)
    #     doc.print(f"Avg return for DQN (4000 epochs) is {returns}")
    #
    # plot_value(q_values, losses, fig_prefix="dqn_2000", title="DQN on Maze (4000 epochs)", doc=doc.table().figure_row())
    #
    # with doc:
    #     def get_Q_rff(B_scale):
    #         return nn.Sequential(
    #             RFF(2, 200, scale=B_scale),
    #             nn.Linear(400, 400),
    #             nn.ReLU(),
    #             nn.Linear(400, 400),
    #             nn.ReLU(),
    #             nn.Linear(400, 400),
    #             nn.ReLU(),
    #             nn.Linear(400, 4),
    #         )
    #
    # doc @ """
    # ## DQN with RFF
    #
    # We can now apply this to DQN and it works right away! Using scale of 5
    # """
    # with doc:
    #     b_scale = 1
    #     Q = get_Q_rff(B_scale=b_scale)
    #     q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, n_epochs=400)
    #     returns = eval_q_policy(Q)
    #
    #     doc.print(f"Avg return for DQN+RFF is {returns}")
    #
    # plot_value(q_values, losses, fig_prefix=f"dqn_rff_{b_scale}", title=f"DQN RFF $\sigma={b_scale}$", doc=doc.table().figure_row())
    # doc.flush()
