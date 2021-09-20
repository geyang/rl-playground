
## Learned Fourier Features

We use stacked, four-layer Learned Fourier Networks (LFN) to fit to a complex value function.

The figure table below shows that with correct scaling, the spectral bias persist across networks
of different width across 8 octaves of latent dimension.

```python
num_states = 200
torch.manual_seed(0)
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/value_iteration.png?ts=316071" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ LFF

Here we plot the value function learned via deep Q Learning (DQN) using a learned random
fourier feature network.

```python
q_values = perform_deep_vi_lff_mlp(states, rewards, dyn_mats, B_scale=8, n_epochs=200)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/dqn_lff_mlp.png?ts=798720" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ RFF

Here we plot the value function learned via deep Q Learning (DQN) using a random
fourier feature network.

```python
q_values = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=10, n_epochs=200)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/dqn_rff_mlp.png?ts=826823" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
