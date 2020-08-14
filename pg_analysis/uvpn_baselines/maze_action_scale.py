import sys
from cmx import doc, CommonMark

# doc = CommonMark(filename="README.md", overwrite=True)
doc @ """
# DQN result on CMaze (discrete) 

"""
prefix = None
# methods = ['dqn', 'ppo', 'sac', 'td3', 'ddpg']
methods = ['dqn']
env_ids = [
    "ge_world:Maze-fixed-discrete-v0",
    # "ge_world:Maze-discrete-v0",
    # "ge_world:CMaze-discrete-v0",
]

# prefix = "geyang/playground/2020/08-02/mujoco_all/01.13.15"
# launch training
if not prefix:
    import gym
    import jaynes
    from firedup.algos.dqn.dqn_her import dqn
    # from firedup.algos.dqn.dqn import dqn
    # from firedup.algos.dqn.dqn_v2 import dqn
    from firedup.algos.ppo.ppo import ppo
    from firedup.algos.sac.sac import sac
    from firedup.algos.td3.td3 import td3
    from firedup.algos.ddpg.ddpg import ddpg
    from pg_experiments import instr

    debug = "pydevd" in sys.modules
    jaynes.config("local" if debug else None)

    for env_id in env_ids:
        for method in methods:
            for act_scale in [0.125, 0.25, 0.5]:
                for seed in [100, 200, 300, 400, 500]:
                    thunk = instr(eval(method),
                                  # env_id="LunarLander-v2",
                                  env_id=env_id,
                                  env_kwargs=dict(act_scale=act_scale),
                                  obs_keys=("x", "goal"),
                                  her_k=1,
                                  max_ep_len=200,
                                  ac_kwargs=dict(hidden_sizes=[128, ] * 3),
                                  gamma=0.99,
                                  seed=seed,
                                  steps_per_epoch=4000,
                                  epochs=500 if method == "ppo" else 200,
                                  _job_prefix="debug" if debug else None,
                                  _job_postfix=f"{env_id.split(':')[-1]}-{method}-{act_scale}")

                    jaynes.run(thunk)

    jaynes.listen()

if __name__ == '__main__':
    # analysis
    import matplotlib.pyplot as plt
    from ml_logger import ML_Logger
    from pg_experiments import RUN
    from pg_analysis import plot_area, COLORS

    loader = ML_Logger(root_dir=RUN.server, prefix=prefix)

    for env_id in env_ids:

        with doc.row():

            plt.figure()
            plt.title(env_id)

            for i, method in enumerate(methods):
                print(f'{env_id}/{method}')
                yKey = "EpRet/mean" if method == 'ppo' else "TestEpRet/mean"
                success = loader.read_metrics("envSteps", yKey, path=f"**/{env_id}/{method}/**/metrics.pkl")
                # print(success.head())
                # env_id = loader.read_params("env_id")
                plot_area(success, "envSteps", yKey, label=method.upper(),
                          color=COLORS[i % len(COLORS)], x_opts={"scale": 1000}, y_opts={"scale": "k"})

            plt.xlim(0, 200_000)
            plt.xlabel('Environment Steps (k)')
            plt.ylabel('Reward')
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            doc.savefig(f"figures/{env_id}_steps.png", zoom="50%", bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.title(env_id)

            for i, method in enumerate(methods):
                print(f'{env_id}/{method}')
                yKey = "EpRet/mean" if method == 'ppo' else "TestEpRet/mean"
                success = loader.read_metrics("time", yKey, path=f"**/{env_id}/{method}/**/metrics.pkl")
                # print(success.head())
                # config = loader.read_params("env_id")
                plot_area(success, "time", yKey, label=method.upper(),
                          color=COLORS[i % len(COLORS)], x_format="timedelta", y_opts={"scale": "k"})

            plt.xlim(0, 1800)
            plt.xlabel('Wall-clock Time')
            plt.ylabel('Reward')
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            doc.savefig(f"figures/{env_id}_wall_clock.png", zoom="50%", bbox_inches="tight")
            plt.close()

    doc.flush()
