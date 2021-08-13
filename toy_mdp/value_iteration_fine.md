
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
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/value_iteration.png?ts=365196" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/value_iteration_loss.png?ts=145594" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ Function Approximator

Here we plot the value function learned via deep Q Learning 
(DQN) using a neural network function approximator.

```python
q_values, losses, avg_returns = perform_deep_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn.png?ts=217168" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_loss.png?ts=905777" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

**But can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
q_values, losses, avg_returns = supervised(states, gt_q_values, dyn_mats, n_epochs=8000)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised.png?ts=127329" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_loss.png?ts=841250" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Now use RFF (supervised)

The same supervised experiment, instantly improve in fit if we 
replace the input layer with RFF embedding.

```python
q_values, losses, avg_returns = supervised_rff(states, gt_q_values, dyn_mats, B_scale=10)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_rff.png?ts=470749" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_rff_loss.png?ts=169244" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with RFF 

We can now apply this to DQN and it works right away! Using scale of 5

```python
q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=10)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_10.png?ts=430713" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_10_loss.png?ts=496197" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN without the Target Q

Setting the target network to off

```python
q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=10, target_freq=None)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_no_target.png?ts=195097" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_no_target_loss.png?ts=894442" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


We can experiment with different scaling $\sigma$

```python
q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=1)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_1.png?ts=890562" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_1_loss.png?ts=573408" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

```python
q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=3)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_3.png?ts=766670" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_3_loss.png?ts=515309" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

```python
q_values, losses, avg_returns = perform_deep_vi_rff(states, rewards, dyn_mats, B_scale=5)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_5.png?ts=806954" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_5_loss.png?ts=523074" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
