
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows that even for simple dynamics, the
value function can be exponentially complex due to recursion.

```python
num_states = 20
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
gt_q_values = np.loadtxt("data/q_values.csv", delimiter=',')
```
| <img style="align-self:center; zoom:0.3;" src="figures/toy_mdp.png?ts=691952" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

**Can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
q_values, losses = supervised(states, gt_q_values, dyn_mats, lr=3e-4)
```
| <img style="align-self:center; zoom:0.3;" src="figures/supervised.png?ts=367884" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="figures/supervised_loss.png?ts=781522" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Supervised, Random Fourier Features

**Can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
for scale in [1, 10, 100]:
    q_values, losses = supervised_rff(states, gt_q_values, dyn_mats, lr=3e-4, rff_scale=scale)
    r = t.figure_row()
    plot_value(states, q_values, losses, f"rff_{scale}", doc=r)
```

| <img style="align-self:center; zoom:0.3;" src="figures/rff_1.png?ts=525478" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="figures/rff_1_loss.png?ts=899625" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img style="align-self:center; zoom:0.3;" src="figures/rff_10.png?ts=706331" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="figures/rff_10_loss.png?ts=082471" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
| <img style="align-self:center; zoom:0.3;" src="figures/rff_100.png?ts=751640" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="figures/rff_100_loss.png?ts=162846" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
