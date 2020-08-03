from cmx import doc, CommonMark

# doc = CommonMark(filename="README.md", overwrite=True)
doc @ """
# All baselines on Cheetah-v2

As mentioned in the Spinup documentation, the PPO 
implementation here is not on-par with SOTA because
it is missing pretty common tricks such as reward 
normalization etc. The SAC, DDPG and TD3 implementations
however are. 

The `mrl` library attains stronger performance.
"""
prefix = None
methods = ['ppo', 'sac', 'td3', 'ddpg']

prefix = "geyang/playground/2020/08-01/cheetah_all/23.14.48"

env_id = "HalfCheetah"

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

    for method in methods:
        for seed in [100, 200, 300, 400, 500]:
            thunk = instr(eval(method),
                          env_id=f"{env_id}-v2",
                          ac_kwargs=dict(hidden_sizes=[128, ] * 3),
                          gamma=0.99,
                          seed=seed,
                          steps_per_epoch=4000,
                          epochs=500 if method == "ppo" else 50,
                          _job_postfix=method)

            jaynes.run(thunk)

    jaynes.listen()

if __name__ == '__main__':
    # analysis
    import matplotlib.pyplot as plt
    from ml_logger import ML_Logger
    from pg_experiments import RUN
    from pg_analysis import plot_area, COLORS

    loader = ML_Logger(log_directory=RUN.server, prefix=prefix)

    with doc.row():

        plt.figure()
        plt.title(env_id)

        for i, method in enumerate(methods):
            yKey = "EpRet/mean" if method == 'ppo' else "TestEpRet/mean"
            success = loader.read_metrics("envSteps", yKey, path=f"**/{method}/**/metrics.pkl")
            # print(success.head())
            # config = loader.read_params("env_id")
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
