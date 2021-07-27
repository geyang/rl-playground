from copy import deepcopy

import numpy as np
import torch.optim
from cmx import doc
from tqdm import trange


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


def perform_deep_vi(states, rewards, dyn_mats, lr=1e-4, gamma=0.9):
    import torch.nn as nn
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        nn.Linear(1, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 2),
    )
    Q_target = deepcopy(Q)

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    rewards = torch.FloatTensor(rewards)
    dyn_mats = torch.FloatTensor(dyn_mats)

    losses = []

    for epoch in trange(400):
        if epoch % 10 == 0:
            Q_target.load_state_dict(Q.state_dict())

        q_max, actions = Q_target(states).max(dim=-1)
        td_target = rewards + gamma * dyn_mats @ q_max
        td_loss = l1(Q(states), td_target.T)
        losses.append(td_loss.detach().numpy())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    return q_values, losses


if __name__ == "__main__":
    doc @ """
    Here is the ground truth value function generated via tabular
    value iteration. It shows even for simple dynamics, the value
    function can be exponentially more complex.
    """
    with doc.table().figure_row() as r:
        from rand_mdp import RandMDP
        from matplotlib import pyplot as plt

        num_states = 100
        mdp = RandMDP(seed=0, option='fixed')
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, loss = perform_vi(states, rewards, dyn_mats)
        plt.plot(states, q_values[0], label="action 1")
        plt.plot(states, q_values[1], label="action 2")
        plt.title("Toy MDP")
        plt.legend()
        plt.xlabel('State [0, 1)')
        plt.ylabel('Value')
        r.savefig(f'figures/toy_mdp.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

        plt.plot(loss)
        plt.hlines(0, 0, len(states), linestyle='--', color='gray')
        plt.title("Residual")
        plt.xlabel('Optimization Steps')
        r.savefig(f'figures/residual.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

    doc @ """
    Here we plot the value function learned via deep Q Learning 
    (DQN) using a neural network function approximator.
    """
    doc.flush()

    with doc.table().figure_row() as r:
        from rand_mdp import RandMDP
        from matplotlib import pyplot as plt

        num_states = 100
        mdp = RandMDP(seed=0, option='fixed')
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, losses = perform_deep_vi(states, rewards, dyn_mats)
        plt.plot(states, q_values[0], label="action 1")
        plt.plot(states, q_values[1], label="action 2")
        plt.title("Toy MDP")
        plt.legend()
        plt.xlabel('State [0, 1)')
        plt.ylabel('Value')
        r.savefig(f'figures/q_learning.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

        plt.plot(losses)
        plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
        plt.title("Residual")
        plt.xlabel('Optimization Steps')
        r.savefig(f'figures/td_loss.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

    doc.flush()
