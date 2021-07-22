from firedup.algos.dqn.dqn_v2 import dqn
from toy_mdp.rand_mdp import RandMDP
import os

seed = 0
save_dir = f'./toy_mdp_analysis/dqn_toy_mdp/{seed}'

os.makedirs(save_dir, exist_ok=True)
env, test_env = RandMDP(option='fixed'), RandMDP(option='fixed')

dqn(env=env,
    test_env=test_env,
    exp_name=f'rand_mdp_fixed_dqn/{seed}',
    ac_kwargs=dict(hidden_sizes=[256, ] * 3),
    gamma=0.99,
    seed=seed,
    steps_per_epoch=4000,
    epochs=500,
    save_dir=save_dir)