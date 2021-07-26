import numpy as np
from cmx import doc



def perform_vi(states, rewards, dyn_mats, gamma=0.9, eps=1e-5):
    # Assume discrete actions and states
    num_actions, num_states = dyn_mats.shape[:2]
    q_values = np.zeros((num_actions, num_states))
    done = False
    deltas = []
    while not done:
        new_q_values = np.zeros_like(q_values)

        for action in range(num_actions):
            new_q_values[action] = rewards[action] + gamma * dyn_mats[action].dot(np.max(q_values, axis=0))

        delta = np.max(np.abs(new_q_values - q_values))
        deltas.append(delta)
        done = (delta < eps)
        q_values = new_q_values

    return q_values, deltas


if __name__ == "__main__":
    with doc, doc.table().figure_row() as r:
        from rand_mdp import RandMDP
        from matplotlib import pyplot as plt

        num_states = 200
        mdp = RandMDP(seed=0, option='fixed')
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, loss = perform_vi(states, rewards, dyn_mats)
        q_values.shape
        plt.plot(states, q_values[0], label="action 1")
        plt.plot(states, q_values[1], label="action 2")
        plt.title("Toy MDP")
        plt.legend()
        plt.xlabel('State [0, 1)')
        plt.ylabel('Value')
        r.savefig(f'figures/toy_mdp.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

        plt.plot(loss)
        plt.title("Residual")
        plt.xlabel('Optimization Steps')
        r.savefig(f'figures/residual.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()
