from cmx import doc, CommonMark

# doc = CommonMark(filename="README.md", overwrite=True)
doc @ """
# Result Summary

"""
prefix = None
methods = ['ppo', 'sac', 'td3']

prefix = "geyang/playground/2020/08-01/cheetah/18.36.22"

env_id = "HalfCheetah"

# launch training
if not prefix:
    import gym
    import jaynes
    from playground.algos.ppo.ppo import ppo
    from playground.algos.sac.sac import sac
    from playground.algos.td3.td3 import td3
    from pg_experiments import instr

    jaynes.config()

    for method in methods:
        for seed in [100, 200, 300]:
            thunk = instr(eval(method),
                          lambda: gym.make(f"{env_id}-v2"),
                          ac_kwargs=dict(hidden_sizes=[64, ] * 2),
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
            success = loader.read_metrics("envSteps", "EpRet/mean",
                                          path=f"**/{method}/**/metrics.pkl")
            # print(success.head())
            # config = loader.read_params("env_id")
            plot_area(success, "envSteps", "EpRet/mean", color=COLORS[i % len(COLORS)],
                      label=method.upper(), x_opts={"scale": 1000}, y_opts={"scale": "k"})

        plt.xlim(0, 200_000)
        plt.xlabel('Environment Steps (k)')
        plt.ylabel('Reward')
        plt.legend(bbox_to_anchor=(1, 0.85))
        plt.tight_layout()
        doc.savefig(f"figures/{env_id}_steps.png", zoom="50%", dpi=120)

        plt.figure()
        plt.title(env_id)

        for i, method in enumerate(methods):
            success = loader.read_metrics("time", "EpRet/mean",
                                          path=f"**/{method}/**/metrics.pkl")
            # print(success.head())
            # config = loader.read_params("env_id")
            plot_area(success, "time", "EpRet/mean", color=COLORS[i % len(COLORS)],
                      label=method.upper(), x_format="timedelta", y_opts={"scale": "k"})

        plt.xlabel('Wall-clock Time')
        plt.ylabel('Reward')
        plt.legend(bbox_to_anchor=(1, 0.85))
        plt.tight_layout()
        doc.savefig(f"figures/{env_id}_wall_clock.png", zoom="50%", dpi=120)

    doc.flush()
