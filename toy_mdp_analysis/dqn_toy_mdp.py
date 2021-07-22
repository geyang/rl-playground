from firedup.algos.dqn.dqn_v2 import dqn
import os

seed = 0
save_dir = f'./toy_mdp_analysis/dqn_toy_mdp/{seed}'

os.makedirs(save_dir, exist_ok=True)

dqn("semi-rand-mdp",
    ac_kwargs=dict(hidden_sizes=[64, ] * 2),
    gamma=0.99,
    seed=seed,
    steps_per_epoch=4000,
    epochs=50,
    save_dir=save_dir)