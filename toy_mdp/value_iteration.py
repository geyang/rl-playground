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


def supervised(states, values, dyn_mats, lr=4e-4, gamma=0.9, n_epochs=100):
    import torch.nn as nn
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
    return q_values, losses


def perform_deep_vi(states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400):
    import torch.nn as nn
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
    return q_values, losses


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
        num_states = 20
        mdp = RandMDP(seed=0, option='fixed')
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, losses = perform_vi(states, rewards, dyn_mats)
    q_values_vi = q_values  # used later
    with doc.table().figure_row() as r:
        plt.plot(states, q_values[0], label="action 1")
        plt.plot(states, q_values[1], label="action 2")
        plt.title("Toy MDP")
        plt.legend()
        plt.xlabel('State [0, 1)')
        plt.ylabel('Value')
        r.savefig(f'figures/toy_mdp.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

        plt.plot(losses)
        plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
        plt.title("Residual")
        plt.xlabel('Optimization Steps')
        r.savefig(f'figures/residual.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

    doc @ """
    ## DQN w/ Function Approximator
    
    Here we plot the value function learned via deep Q Learning 
    (DQN) using a neural network function approximator.
    """

    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    mdp = RandMDP(seed=0, option='fixed')
    states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
    with doc:
        q_values, losses = perform_deep_vi(states, rewards, dyn_mats)
    with doc.table().figure_row() as r:
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

    doc @ """
    ## A Supervised Baseline
    
    **But can the function learn these value functions?** As it turned out, no.
    Even with a supervised learning objective, the learned value function is
    not able to produce a good approximation of the value landscape. Not
    with 20 states, and even less so with 200.
    """
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    num_states = 20
    mdp = RandMDP(seed=0, option='fixed')
    states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
    with doc:
        q_values, losses = supervised(states, q_values_vi, dyn_mats)
    with doc.table().figure_row() as r:
        plt.plot(states, q_values[0], label="action 1")
        plt.plot(states, q_values[1], label="action 2")
        plt.title("Toy MDP")
        plt.legend()
        plt.xlabel('State [0, 1)')
        plt.ylabel('Value')
        r.savefig(f'figures/supervised.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

        plt.plot(losses)
        plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
        plt.title("Loss")
        plt.xlabel('Optimization Steps')
        r.savefig(f'figures/supervised_loss.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()
    doc.flush()
