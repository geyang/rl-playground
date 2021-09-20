
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
torch.manual_seed(0)
rewards, dones, dyn_mats = get_discrete_mdp()
```

## DQN w/ Function Approximator

Here we plot the value function learned via deep Q Learning
(DQN) using a neural network function approximator.

```python
def get_Q_mlp():
    return nn.Sequential(
        Lambda(lambda x: x / 255),
        # RFF(3, 8, 3, stride=1),
        nn.Conv2d(3, 8, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1),
        nn.ReLU(),
        View(8*58*58),
        nn.Linear(8*58*58, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 4),
    )


Q = get_Q_mlp()
q_values, losses = perform_deep_vi(Q, rewards, dyn_mats, dones, n_epochs=1500)
returns = eval_q_policy(Q)
doc.print(f"Avg return for DQN is {returns}")
```

```
Avg return for DQN is -0.4000000000000002
```
