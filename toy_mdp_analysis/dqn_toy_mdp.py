from firedup.algos.dqn.dqn_v2 import dqn
from toy_mdp.rand_mdp import RandMDP
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument('--seeds', nargs='+', type=int, help='seeds', required=True)
    args = parser.parse_args()
    device='cuda'

    for seed in args.seeds:
        save_dir = f'./toy_mdp_analysis/dqn_toy_mdp/{seed}'
        os.makedirs(save_dir, exist_ok=True)
        env, test_env = RandMDP(option='fixed'), RandMDP(option='fixed')
        dqn(env=env,
            test_env=test_env,
            exp_name=f'rand_mdp_fixed_dqn/{seed}',
            ac_kwargs=dict(hidden_sizes=[512,], fourier_features=True, fourier_size=8, fourier_sigma=1, device=device),
            gamma=0.9,
            ep_limit=10,
            lr=1e-3,
            replay_size=int(1e4),
            batch_size=128,
            target_update_interval=500,
            min_replay_history=0,
            update_interval=1,
            seed=seed,
            steps_per_epoch=1000,
            epochs=3000,
            device=device,
            save_dir=save_dir,
        )