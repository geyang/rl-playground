from firedup.algos.dqn.dqn_v2 import dqn
from toy_mdp.rand_mdp import RandMDP
import os

seeds = list(range(5))

for seed in seeds:
    save_dir = f'./toy_mdp_analysis/dqn_toy_mdp/{seed}'
    os.makedirs(save_dir, exist_ok=True)
    env, test_env = RandMDP(option='fixed'), RandMDP(option='fixed')
    dqn(env=env,
        test_env=test_env,
        exp_name=f'rand_mdp_fixed_dqn/{seed}',
        ac_kwargs=dict(hidden_sizes=[128, ] * 2),
        gamma=0.9,
        lr=1e-3,
        replay_size=int(1e4),
        batch_size=128,
        target_update_interval=50,
        min_replay_history=0,
        update_interval=1,
        seed=seed,
        steps_per_epoch=1000,
        epochs=300,
        save_dir=save_dir)