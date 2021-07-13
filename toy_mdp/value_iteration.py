import numpy as np

def perform_vi(states, rewards, dyn_mats, gamma=0.9, eps=1e-5):
    # Assume discrete actions and states
    num_actions, num_states = dyn_mats.shape[:2]
    q_values = np.zeros((num_actions, num_states))
    done = False
    while not done:
        new_q_values = np.zeros_like(q_values)

        for action in range(num_actions):
            new_q_values[action] = rewards[action] + gamma * dyn_mats[action].dot(np.max(q_values, axis=0))

        delta = np.max(np.abs(new_q_values - q_values))
        done = (delta < eps)
        q_values = new_q_values

    return q_values

if __name__ == "__main__":
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt
    num_states = 1000
    mdp = RandMDP(seed=0, option='fixed')
    states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
    q_values = perform_vi(states, rewards, dyn_mats)
    plt.plot(states, q_values[0])
    plt.plot(states, q_values[1])
    plt.show()
