
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
torch.manual_seed(0)
rewards, dones, dyn_mats = get_discrete_mdp()
```
