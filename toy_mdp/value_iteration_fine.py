import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange


def plot_value(states, q_values, losses, fig_prefix, title=None, doc=doc):
    plt.plot(states, q_values[0], label="action 1")
    plt.plot(states, q_values[1], label="action 2")
    if title:
        plt.title(title)
    plt.legend()
    plt.xlabel('State [0, 1)')
    plt.ylabel('Value')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()

    plt.plot(losses)
    plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
    plt.title("Loss")
    plt.xlabel('Optimization Steps')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}_loss.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()


def perform_vi(states, rewards, dyn_mats, gamma=0.9, eps=1e-5):
    # Assume discrete actions and states
    q_values = np.zeros(dyn_mats.shape[:2])

    deltas = []
    while not deltas or deltas[-1] >= eps:
        old = q_values
        q_max = q_values.max(axis=0)
        q_values = rewards + gamma * dyn_mats @ q_max

        deltas.append(np.abs(old - q_values).max())

    return q_values, deltas


def supervised(states, values, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=100):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        nn.Linear(1, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    values = torch.FloatTensor(values)

    losses = []

    for epoch in trange(n_epochs + 1):
        values_bar = Q(states)
        loss = l1(values_bar, values.T)
        losses.append(loss.detach().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()

    q_values = values_bar.T.detach().numpy()
    avg_returns = eval_q_policy(Q)
    return q_values, losses, avg_returns


def perform_deep_vi(states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        nn.Linear(1, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )
    Q_target = deepcopy(Q)

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    rewards = torch.FloatTensor(rewards)
    dyn_mats = torch.FloatTensor(dyn_mats)

    losses = []

    for epoch in trange(n_epochs + 1):
        if epoch % 1 == 0:
            Q_target.load_state_dict(Q.state_dict())

        q_max, actions = Q_target(states).max(dim=-1)
        td_target = rewards + gamma * dyn_mats @ q_max
        td_loss = l1(Q(states), td_target.T)
        losses.append(td_loss.detach().numpy())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    avg_returns = eval_q_policy(Q)
    return q_values, losses, avg_returns


class RFF(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = mapping_size * 2
        self.B = torch.normal(0, scale, size=(mapping_size, self.input_dim))
        self.B.requires_grad = False

    def forward(self, x):
        return torch.cat([torch.cos(2 * np.pi * x @ self.B.T),
                          torch.sin(2 * np.pi * x @ self.B.T)], dim=1)


def supervised_rff(states, values, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=100, B_scale=1):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        RFF(1, 200, scale=B_scale),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    values = torch.FloatTensor(values)

    losses = []

    for epoch in trange(n_epochs + 1):
        values_bar = Q(states)
        loss = l1(values_bar, values.T)
        losses.append(loss.detach().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()

    q_values = values_bar.T.detach().numpy()
    avg_returns = eval_q_policy(Q)
    return q_values, losses, avg_returns


def perform_deep_vi_rff(states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, B_scale=1, target_freq=1):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        RFF(1, 200, scale=B_scale),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )
    Q_target = deepcopy(Q) if target_freq else Q

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
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
    avg_returns = eval_q_policy(Q)
    return q_values, losses, avg_returns

def eval_q_policy(q, num_eval=100):
    """Assumes discrete action such that policy is derived by argmax a Q(s,a)"""
    from rand_mdp import RandMDP
    torch.manual_seed(0)
    env = RandMDP(seed=0, option='fixed')
    returns = []

    for i in range(num_eval):
        done = False
        obs = env.reset()
        total_rew = 0
        while not done:
            obs = torch.FloatTensor(obs).unsqueeze(-1)
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
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    with doc:
        num_states = 200
        torch.manual_seed(0)
        mdp = RandMDP(seed=0, option='fixed')
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, losses = perform_vi(states, rewards, dyn_mats)

    os.makedirs('data', exist_ok=True)
    np.savetxt("data/states.csv", states, delimiter=',')
    np.savetxt("data/rewards.csv", states, delimiter=',')
    np.savetxt("data/q_values.csv", q_values, delimiter=',')

    gt_q_values = q_values  # used later

    plot_value(states, q_values, losses, fig_prefix="value_iteration",
               title="Value Iteration on Toy MDP", doc=doc.table().figure_row())

    doc @ """
    ## DQN w/ Function Approximator
    
    Here we plot the value function learned via deep Q Learning 
    (DQN) using a neural network function approximator.
    """

    with doc:
        q_values, losses, avg_returns = perform_deep_vi(states, rewards, dyn_mats)

    print(f"Avg return for DQN is {avg_returns}")
    plot_value(states, q_values, losses, fig_prefix="dqn",
               title="DQN on Toy MDP", doc=doc.table().figure_row())

    doc @ """
    ## A Supervised Baseline
    
    **But can the function learn these value functions?** As it turned out, no.
    Even with a supervised learning objective, the learned value function is
    not able to produce a good approximation of the value landscape. Not
    with 20 states, and even less so with 200.
    """
    with doc:
        q_values, losses, avg_returns = supervised(states, gt_q_values, dyn_mats, n_epochs=8000)

    print(f"Avg return for NN+sup is {avg_returns}")
    plot_value(states, q_values, losses, fig_prefix="supervised",
               title="Supervised Value Function", doc=doc.table().figure_row())

    doc @ """
    ## Now use RFF (supervised)
    
    The same supervised experiment, instantly improve in fit if we 
    replace the input layer with RFF embedding.
    """
    with doc:
        q_values, losses, avg_returns = supervised_rff(states, gt_q_values, dyn_mats, B_scale=10)

    print(f"Avg return for NN+RFF+sup is {avg_returns}")
    plot_value(states, q_values, losses, fig_prefix="supervised_rff",
               title=f"RFF Supervised {10}", doc=doc.table().figure_row())
    doc @ """
    ## DQN with RFF 
    
    We can now apply this to DQN and it works right away! Using scale of 5
    """
    with doc:
        q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=10)

    print(f"Avg return for DQN+RFF is {avg_returns}")
    plot_value(states, q_values, losses, fig_prefix=f"dqn_rff_{10}",
               title=f"DQN RFF $\sigma={10}$", doc=doc.table().figure_row())

    doc @ """
    ## DQN without the Target Q
    
    Setting the target network to off
    """
    with doc:
        q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=10, target_freq=None)
    print(f"Avg return for DQN+RFF-tgt is {avg_returns}")
    plot_value(states, q_values, losses, fig_prefix=f"dqn_rff_no_target",
               title=f"DQN RFF No Target", doc=doc.table().figure_row())

    doc @ """
    We can experiment with different scaling $\sigma$
    """
    with doc:
        q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=1)
    print(f"Avg return for DQN+RFF (sigma 1) is {avg_returns}")
    plot_value(states, q_values, losses, fig_prefix=f"dqn_rff_{1}",
               title=f"DQN RFF $\sigma={1}$", doc=doc.table().figure_row())
    with doc:
        q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=3)
    print(f"Avg return for DQN+RFF (sigma 3) is {avg_returns}")
    plot_value(states, q_values, losses, fig_prefix=f"dqn_rff_{3}",
               title=f"DQN RFF $\sigma={3}$", doc=doc.table().figure_row())

    with doc:
        q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=5)
    print(f"Avg return for DQN+RFF (sigma 5) is {avg_returns}")
    plot_value(states, q_values, losses, fig_prefix=f"dqn_rff_{5}",
               title=f"DQN RFF $\sigma={5}$", doc=doc.table().figure_row())
    doc.flush()
