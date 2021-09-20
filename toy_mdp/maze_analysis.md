
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
torch.manual_seed(0)
rewards, dones, dyn_mats = get_discrete_mdp()
q_values, losses = perform_vi(rewards, dyn_mats, dones)
```
| <img style="align-self:center; zoom:0.3;" src="maze_analysis/value_iteration.png?ts=950441" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="maze_analysis/value_iteration_loss.png?ts=195565" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

```python
mean_reward = eval_q_policy(Q_table_wrapper(q_values))
doc.print(f"Return with ground truth q function is {mean_reward}")
```

```
Return with ground truth q function is 0.948
```

## DQN w/ Function Approximator

Here we plot the value function learned via deep Q Learning
(DQN) using a neural network function approximator.

```python
def get_Q_mlp():
    return nn.Sequential(
        nn.Linear(2, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 4),
    )


Q = get_Q_mlp()
q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, dones, n_epochs=400)
returns = eval_q_policy(Q)
doc.print(f"Avg return for DQN is {returns}")
```

```
Avg return for DQN is -0.4000000000000002
```
| <img style="align-self:center; zoom:0.3;" src="maze_analysis/dqn.png?ts=854325" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="maze_analysis/dqn_loss.png?ts=089228" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

```python
Q = get_Q_mlp()
q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, dones, n_epochs=4000)
returns = eval_q_policy(Q)
doc.print(f"Avg return for DQN (4000 epochs) is {returns}")
```

```
Avg return for DQN (4000 epochs) is 0.948
```
| <img style="align-self:center; zoom:0.3;" src="maze_analysis/dqn_2000.png?ts=376889" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="maze_analysis/dqn_2000_loss.png?ts=648260" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

```python
def get_Q_rff(B_scale):
    return nn.Sequential(
        RFF(2, 200, scale=B_scale),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 4),
    )
```

## DQN with RFF

We can now apply this to DQN and it works right away! Using scale of 5

```python
b_scale = 1
Q = get_Q_rff(B_scale=b_scale)
q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, dones, n_epochs=400)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF is {returns}")
```

```
Avg return for DQN+RFF is 0.948
```
| <img style="align-self:center; zoom:0.3;" src="maze_analysis/dqn_rff_1.png?ts=407617" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="maze_analysis/dqn_rff_1_loss.png?ts=703701" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
