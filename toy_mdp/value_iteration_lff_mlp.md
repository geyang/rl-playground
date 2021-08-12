
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
num_states = 200
torch.manual_seed(0)
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/value_iteration.png?ts=134374" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/value_iteration_loss.png?ts=974475" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


# Supervised Learning with Learned Random Fourier Features (LFF)

The random matrix simply does not update that much!

```python
q_values, losses, B_stds, B_means = supervised_lff_mlp(states, gt_q_values, B_scale=8, n_epochs=100)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/supervised_lff_mlp.png?ts=359341" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/supervised_lff_mlp_loss.png?ts=042359" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/supervised_lff_mlp_stddev.png?ts=446509" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/supervised_lff_mlp_mean.png?ts=761717" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

## DQN w/ LFF

Here we plot the value function learned via deep Q Learning (DQN) using a learned random
fourier feature network.

```python
q_values, losses, B_stds, B_means = perform_deep_vi_lff_mlp(states, rewards, dyn_mats, B_scale=8, n_epochs=100)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/dqn_lff_mlp.png?ts=908633" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/dqn_lff_mlp_loss.png?ts=279575" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/dqn_lff_mlp_stddev.png?ts=559000" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff_mlp/dqn_lff_mlp_mean.png?ts=055967" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
