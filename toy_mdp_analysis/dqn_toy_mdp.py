from firedup.algos.dqn.dqn_v2 import dqn
from toy_mdp.rand_mdp import RandMDP
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config')
    parser.add_argument('--seeds', nargs='+', type=int, help='seeds', required=True)
    parser.add_argument('--fft', action='store_true')
    parser.add_argument('--no_target', action='store_true')
    args = parser.parse_args()
    device='cuda'

    if args.fft:
        fourier_features = True
        fourier_size = 8
        fourier_sigma = 10
        exp_name = 'rand_mdp_fixed_dqn_fft'
        save_dir = 'toy_mdp_analysis/dqn_toy_mdp_fft'
    else:
        fourier_features = False
        fourier_size = -1
        fourier_sigma = -1
        exp_name = 'rand_mdp_fixed_dqn'
        save_dir = 'toy_mdp_analysis/dqn_toy_mdp'

    if args.no_target:
        exp_name += '_no_tgt'
        save_dir += '_no_tgt'

    for seed in args.seeds:
        save_dir = f'./{save_dir}/{seed}'
        os.makedirs(save_dir, exist_ok=True)
        env, test_env = RandMDP(option='fixed'), RandMDP(option='fixed')
        dqn(env=env,
            test_env=test_env,
            exp_name=f'{exp_name}/{seed}',
            ac_kwargs=dict(hidden_sizes=[512,], fourier_features=fourier_features, fourier_size=fourier_size, fourier_sigma=fourier_sigma, device=device),
            gamma=0.9,
            ep_limit=10,
            lr=1e-4,
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
            no_target=args.no_target,
        )