
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
num_states = 20
torch.manual_seed(0)
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/value_iteration.png?ts=271940" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/value_iteration_loss.png?ts=055160" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ Function Approximator

Here we plot the value function learned via deep Q Learning 
(DQN) using a neural network function approximator.

```python
q_values, losses = perform_deep_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/dqn.png?ts=829883" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/dqn_loss.png?ts=365321" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

**But can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
q_values, losses = supervised(states, gt_q_values, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/supervised.png?ts=513242" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/supervised_loss.png?ts=061016" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Now use RFF (supervised)

The same supervised experiment, instantly improve in fit if we 
replace the input layer with RFF embedding.

```python
q_values, losses = supervised_rff(states, gt_q_values, dyn_mats, B_scale=10)
```
| <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/supervised_rff.png?ts=317298" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/supervised_rff_loss.png?ts=037601" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with RFF 

We can now apply this to DQN and it works right away!

```python
q_values, losses = perform_deep_vi_rff(states, rewards, dyn_mats, n_epochs=500, B_scale=10)
```
| <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/dqn_rff.png?ts=462228" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/dqn_rff_loss.png?ts=003101" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with RFF without Target

Try removing the target network

```python
q_values, losses = perform_deep_vi_rff(states, rewards, dyn_mats, n_epochs=500, B_scale=10,
                                       target_freq=None)
```
| <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/dqn_rff_no_target.png?ts=869640" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="/Users/ge/mit/playground/toy_mdp/value_iteration/figures/dqn_rff_no_target_loss.png?ts=347893" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
