from cmx import doc, CommonMark

# doc = CommonMark(filename="README.md", overwrite=True)
from firedup.wrappers.flat_goal import FlatGoalEnv

doc @ """
# Baselines on All Fetch Tasks

How well does this learn on the fetch tasks? Without HER, 
fetch tasks should not work well. We use the dense reward
version for this reason.

## To-dos

- [ ] add HER
- [ ] add vectorized environment

At the same time could try `mrl` baseline on these environments.
"""
prefix = None
# methods = ['ppo', 'sac', 'td3', 'ddpg']
# env_ids = ["FetchSlide-v1", "FetchPickAndPlace-v1", "FetchReach-v1", "FetchPush-v1", "FetchSlideDense-v1",
#            "FetchPickAndPlaceDense-v1", "FetchReachDense-v1", "FetchPushDense-v1"]
# seeds = [100, 200, 300, 400, 500]

methods = ['sac', 'td3']
env_ids = ["FetchSlideDense-v1", "FetchPickAndPlaceDense-v1", "FetchReachDense-v1", "FetchPushDense-v1"]
seeds = [100, ]

# prefix = "geyang/playground/2020/08-01/mujoco_all/20.23.21"

# launch training
if not prefix:
    import gym
    import jaynes
    from firedup.algos.ppo.ppo import ppo
    from firedup.algos.sac.sac import sac
    from firedup.algos.td3.td3 import td3
    from firedup.algos.ddpg.ddpg import ddpg
    from pg_experiments import instr

    debug = False
    jaynes.config("local" if debug else None)

    for env_id in env_ids:
        for method in methods:
            for seed in seeds:
                thunk = instr(eval(method),
                              wrappers=(FlatGoalEnv,),
                              env_id=f"{env_id}",
                              ac_kwargs=dict(hidden_sizes=[128, ] * 3),
                              gamma=0.99,
                              seed=seed,
                              steps_per_epoch=4000,
                              epochs=500 if method == "ppo" else 50,
                              _job_postfix="debug" if debug else f"{env_id}/{method}")

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
                yKey = "EpRet/mean" if method == 'ppo' else "TestEpRet/mean"
                success = loader.read_metrics("envSteps", yKey, path=f"**/{method}/**/metrics.pkl")
                # print(success.head())
                env_id = loader.read_params("env_id")
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
                yKey = "EpRet/mean" if method == 'ppo' else "TestEpRet/mean"
                success = loader.read_metrics("time", yKey, path=f"**/{method}/**/metrics.pkl")
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
