import os

import numpy as np
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
    doc.savefig(f'{__file__[:-3]}/figures/{fig_prefix}.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()

    plt.plot(losses)
    plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
    plt.title("Loss")
    plt.xlabel('Optimization Steps')
    doc.savefig(f'{__file__[:-3]}/figures/{fig_prefix}_loss.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()


def supervised(xs, ys, lr=4e-4, n_epochs=100, batch_size=None):
    # Ge: need to initialize the Q function at zero
    f = nn.Sequential(
        nn.Linear(1, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 1),
    )

    optim = torch.optim.RMSprop(f.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    xs = torch.FloatTensor(xs).unsqueeze(-1)
    ys = torch.FloatTensor(ys).unsqueeze(-1)

    losses = []

    for epoch in trange(n_epochs + 1):
        values_bar = f(xs)
        loss = l1(values_bar, ys)
        losses.append(loss.detach().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()

    y_bars = values_bar.detach().squeeze().numpy().T
    return y_bars, losses


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


def supervised_rff(xs, ys, lr=4e-4, n_epochs=100, B_scale=1, batch_size=None):
    # Ge: need to initialize the Q function at zero
    f = nn.Sequential(
        RFF(1, 200, B_scale),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 1),
    )

    optim = torch.optim.RMSprop(f.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    xs = torch.FloatTensor(xs).unsqueeze(-1)
    ys = torch.FloatTensor(ys).unsqueeze(-1)

    losses = []

    for epoch in trange(n_epochs + 1):
        values_bar = f(xs)
        loss = l1(values_bar, ys)
        losses.append(loss.detach().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()

    y_bars = values_bar.detach().squeeze().numpy().T
    return y_bars, losses


if __name__ == "__main__":
    doc @ """
    ## Spectral Perspectives on  Fit
    """
    import torch
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    with doc:
        colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247']

        np.random.seed(0)
        torch.random.manual_seed(0)

        xs = np.linspace(0, 1, 1001)
        ys = np.stack([np.sin(np.random.random() + 2 * np.pi * k * xs) for k in range(5, 55, 5)]).sum(axis=0)

    with doc @ """We can visualize the data:""", doc.table().figure_row() as r:
        plt.plot(xs, ys, label="Data")
        ys_bars, losses = supervised(xs, ys)
        ys_bars_rff, losses_rff = supervised_rff(xs, ys, B_scale=50)

        plt.plot(xs, ys_bars, label="MLP")
        plt.plot(xs, ys_bars_rff, label="RFF")
        plt.legend()
        r.savefig(__file__[:-3] + '/fit.png', title="Fit", zoom=0.3)
        plt.close()

        plt.plot(losses, label="MLP")
        plt.plot(losses_rff, label="RFF")
        plt.legend()
        plt.ylim(0, 2)
        r.savefig(__file__[:-3] + '/loss.png', title="loss", zoom=0.3)

    exit()

    with doc:
        num_states = 20
        torch.manual_seed(0)
        mdp = RandMDP(seed=0, option='fixed')
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, losses = perform_vi(states, rewards, dyn_mats)

    os.makedirs('data', exist_ok=True)
    np.savetxt("data/q_values.csv", q_values, delimiter=',')

    gt_q_values = q_values  # used later

    plot_value(states, q_values, losses, fig_prefix="value_iteration",
               title="Value Iteration on Toy MDP", doc=doc.table().figure_row())

    doc @ """
    ## A Supervised Baseline
    
    **But can the function learn these value functions?** As it turned out, no.
    Even with a supervised learning objective, the learned value function is
    not able to produce a good approximation of the value landscape. Not
    with 20 states, and even less so with 200.
    """
    with doc:
        q_values, losses = supervised(states, gt_q_values, dyn_mats)

    plot_value(states, q_values, losses, fig_prefix="supervised",
               title="Supervised Value Function", doc=doc.table().figure_row())

    doc @ """
    ## Now use RFF (supervised)
    
    The same supervised experiment, instantly improve in fit if we 
    replace the input layer with RFF embedding.
    """
    with doc:
        q_values, losses = supervised_over_param(states, gt_q_values, dyn_mats, B_scale=10)

    plot_value(states, q_values, losses, fig_prefix="supervised_over_param",
               title=f"RFF Supervised {10}", doc=doc.table().figure_row())
    doc.flush()
