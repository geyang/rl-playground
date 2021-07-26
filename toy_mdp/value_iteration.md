```python
from rand_mdp import RandMDP
from matplotlib import pyplot as plt

num_states = 200
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, loss = perform_vi(states, rewards, dyn_mats)
q_values.shape
plt.plot(states, q_values[0], label="action 1")
plt.plot(states, q_values[1], label="action 2")
plt.title("Toy MDP")
plt.legend()
plt.xlabel('State [0, 1)')
plt.ylabel('Value')
r.savefig(f'figures/toy_mdp.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
plt.close()

plt.plot(loss)
plt.title("Residual")
plt.xlabel('Optimization Steps')
r.savefig(f'figures/residual.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
plt.close()
```

| <img style="align-self:center; zoom:0.3;" src="figures/toy_mdp.png?ts=958105" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="figures/residual.png?ts=544417" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------:|
