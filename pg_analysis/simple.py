from cmx import CommonMark

doc = CommonMark(filename="README.md", overwrite=True)
doc @ """
# Result Summary

"""
prefix = "geyang/playground/2020/08-01/cheetah/01.40.12"

# launch training
if not prefix:
    import gym
    import jaynes
    # from playground.algos.ppo.ppo import ppo
    from playground.algos.sac.sac import sac
    from pg_experiments import instr

    jaynes.config()

    for seed in [100, 200, 300]:
        thunk = instr(sac,
                      lambda: gym.make("HalfCheetah-v2"),
                      ac_kwargs=dict(hidden_sizes=[64, ] * 2),
                      gamma=0.99,
                      seed=seed,
                      steps_per_epoch=4000,
                      epochs=50,
                      _job_prefix="sac",
                      )
        jaynes.run(thunk)

    jaynes.listen()

if __name__ == '__main__':
    # analysis
    import matplotlib.pyplot as plt
    from ml_logger import ML_Logger
    from pg_experiments import RUN
    from pg_analysis import plot_area, COLORS

    loader = ML_Logger(root_dir=RUN.server, prefix=prefix)

    plt.figure(figsize=(4.1, 2.8))
    plt.title("Half Cheetah")
    for i, method in enumerate(['ppo', 'sac', 'td3']):
        success = loader.read_metrics("envSteps", "EpRet/mean", path=f"**/{method}/**/metrics.pkl")
        # config = loader.read_params("env_id")
        plot_area(success, "envSteps", "EpRet/mean", color=COLORS[i % len(COLORS)],  label=method)

    plt.xlim(0, 200_000)
    doc.savefig("figures/results.png", zoom="50%", dpi=120)
    doc.flush()
