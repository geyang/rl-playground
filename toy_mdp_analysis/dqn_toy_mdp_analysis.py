import numpy as np
import torch
from matplotlib import pyplot as plt
from firedup.algos.dqn import core
from toy_mdp.rand_mdp import RandMDP

seed = 0
load_path = f'./toy_mdp_analysis/dqn_toy_mdp/{seed}/state.pt'
device='cpu'
env = RandMDP(option='fixed')
ac_kwargs=dict(hidden_sizes=[512,], action_space=env.action_space, fourier_features=False, fourier_size=8, fourier_sigma=10, device=device)
obs_dim = env.observation_space.shape[0]
q_net = core.QMlp(in_features=obs_dim, **ac_kwargs)

state_dict = torch.load(load_path)
q_net.load_state_dict(state_dict['q_net'])

states = np.linspace(0, 1, 1000)
q_vals = np.zeros((2, 1000))

for (i, state) in enumerate(states):
    q_val = q_net(torch.FloatTensor([[state]]))
    q_val = q_val.detach().cpu().numpy()
    q_vals[:,i] = q_val

plt.plot(states, q_vals[0])
plt.plot(states, q_vals[1])
plt.show()