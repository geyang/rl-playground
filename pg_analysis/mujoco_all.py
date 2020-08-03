from cmx import doc, CommonMark

# doc = CommonMark(filename="README.md", overwrite=True)
doc @ """
# Baselines on All gym MuJoCo Tasks

This is still worse than the benchmark result from 
the spinup code base. For next steps:

- [ ] find the exact parameters used for training
- [ ] run later tonight.

Should be able to reproduce the results tomorrow.
"""
prefix = None
methods = ['ppo', 'sac', 'td3', 'ddpg']
env_ids = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2", "Ant-v2"]

prefix = "geyang/playground/2020/08-02/mujoco_all/01.13.15"

# launch training
if not prefix:
    import gym
    import jaynes
    from playground.algos.ppo.ppo import ppo
    from playground.algos.sac.sac import sac
    from playground.algos.td3.td3 import td3
    from playground.algos.ddpg.ddpg import ddpg
    from pg_experiments import instr

    jaynes.config()

    for env_id in env_ids:
        for method in methods:
            for seed in [100, 200, 300, 400, 500]:
                thunk = instr(eval(method),
                              env_id=f"{env_id}",
                              ac_kwargs=dict(hidden_sizes=[128, ] * 3),
                              gamma=0.99,
                              seed=seed,
                              steps_per_epoch=4000,
                              epochs=500 if method == "ppo" else 50,
                              _job_postfix=f"{env_id}/{method}")

                jaynes.run(thunk)

    jaynes.listen()

if __name__ == '__main__':
    # analysis
    import matplotlib.pyplot as plt
    from ml_logger import ML_Logger
    from pg_experiments import RUN
    from pg_analysis import plot_area, COLORS

    loader = ML_Logger(log_directory=RUN.server, prefix=prefix)

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
