from cmx import doc, CommonMark

# doc = CommonMark(filename="README.md", overwrite=True)
doc @ """
# PPO Performance with Parallelization

"""
prefix = None
# methods = ['ppo', 'sac', 'td3']
method = "ppo"
n_cpus = [1, 2, 4, 8, 16]
# n_cpus = [16]

# prefix = "geyang/playground/2020/08-01/cheetah/01.40.12"
prefix = "geyang/playground/2020/08-01/cheetah_ppo_mpi/16.33.14"

env_id = "HalfCheetah"

# launch training
if not prefix:
    import gym
    import jaynes
    from playground.algos.ppo.ppo import ppo
    from pg_experiments import instr

    # n_cpu = 8
    for n_cpu in n_cpus:
        r_conf = {
            'n_cpu': n_cpu * 2,
            'mem': f"{n_cpu}G",
            'entry_script': f"mpirun -np {n_cpu} --oversubscribe /pkgs/anaconda3/bin/python -u -m jaynes.entry"}
        jaynes.config(runner=r_conf)
        for seed in [100, 200, 300]:
            thunk = instr(ppo,
                          lambda: gym.make(f"{env_id}-v2"),
                          ac_kwargs=dict(hidden_sizes=[64, ] * 2),
                          gamma=0.99,
                          seed=seed,
                          steps_per_epoch=2000,
                          epochs=500,
                          _job_postfix=method + f'-c{n_cpu}'
                          )

            jaynes.run(thunk)

    jaynes.listen()

if __name__ == '__main__':
    # analysis
    import matplotlib.pyplot as plt
    from ml_logger import ML_Logger
    from pg_experiments import RUN
    from pg_analysis import plot_area, COLORS

    loader = ML_Logger(log_directory=RUN.server, prefix=prefix)

    with doc.row(styles=dict(width="100%")):

        plt.figure()
        plt.title(env_id)

        for i, n_cpu in enumerate(n_cpus):
            success = loader.read_metrics("envSteps", "EpRet/mean",
                                          path=f"{method}-c{n_cpu}/**/metrics.pkl")
            # config = loader.read_params("env_id")
            plot_area(success, "envSteps", "EpRet/mean",
                      label=f"cpu {n_cpu}",
                      color=COLORS[i % len(COLORS)],
                      x_opts={"scale": "M"}, y_opts={"scale": "k"})

        # plt.xlim(0, 200_000)
        plt.xlabel('Environment Steps (M)')
        plt.ylabel('Reward')
        plt.legend(bbox_to_anchor=(1, 0.85))
        plt.tight_layout()
        doc.savefig(f"figures/{env_id}_mpi_steps.png", zoom="50%")

        plt.figure()
        plt.title(env_id)

        for i, n_cpu in enumerate(n_cpus):
            success = loader.read_metrics("time", "EpRet/mean",
                                          path=f"{method}-c{n_cpu}/**/metrics.pkl")
            # config = loader.read_params("env_id")
            plot_area(success, "time", "EpRet/mean",
                      label=f"cpu {n_cpu}",
                      color=COLORS[i % len(COLORS)],
                      x_format="timedelta", y_opts={"scale": "k"})

        plt.xlabel('Wall-clock Time')
        plt.ylabel('Reward')
        plt.legend(bbox_to_anchor=(1, 0.85))
        plt.tight_layout()
        doc.savefig(f"figures/{env_id}_mpi_wall_clock.png", zoom="50%")

        plt.figure()
        plt.title(env_id)

        for i, n_cpu in enumerate(n_cpus):
            success = loader.read_metrics("epoch", "EpRet/mean",
                                          path=f"{method}-c{n_cpu}/**/metrics.pkl")
            # config = loader.read_params("env_id")
            plot_area(success, "epoch", "EpRet/mean",
                      label=f"cpu {n_cpu}",
                      color=COLORS[i % len(COLORS)],
                      y_opts={"scale": "k"})

        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.legend(bbox_to_anchor=(1, 0.85))
        plt.tight_layout()
        doc.savefig(f"figures/{env_id}_mpi_epoch.png", zoom="50%")

    doc.flush()
