import sys
from cmx import doc
from ml_logger import logger

doc @ """
# DQN result on Maze (discrete) 
"""
with doc:
    env_ids = [
        "ge_world:Maze-fixed-discrete-v0",
        "ge_world:Maze-discrete-v0",
        "ge_world:CMaze-discrete-v0",
        "ge_world:HMaze-discrete-v0",
    ]
    short_names = [d.split(':')[-1].replace("-discrete", "") for d in env_ids]
    prefix = None

if __name__ == '__main__' and prefix:
    doc @ f"""
    Experiment: [[{prefix.split("/")[-2]}]](http://localhost:3001{prefix})
    """
    import matplotlib.pyplot as plt
    from ml_logger import ML_Logger
    from pg_experiments import RUN
    from pg_analysis import plot_area, COLORS

    loader = ML_Logger(root_dir=RUN.server, prefix=prefix)

    method = "dqn"
    i = 0
    with doc.row():
        for env_id, name in zip(env_ids, short_names):
            plt.figure(figsize=(4.5, 2.8))
            plt.title(name)
            xKey = "__timestamp"
            yKey = "test/success/mean"
            success = loader.read_metrics(xKey, yKey, path=f"**/{name}/**/metrics.pkl")
            plot_area(success, xKey, yKey, label=method.upper(),
                      color=COLORS[i % len(COLORS)], x_format="timedelta", y_opts={"scale": "pc"})

            plt.xlabel('Wallclock Time')
            plt.ylabel('Success')
            # plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            doc.savefig(f"figures/loose_goal/{name}_steps.png", zoom="50%", bbox_inches='tight')
            plt.close()

            # plt.figure()
            # plt.title(env_id)
            #
            # for i, method in enumerate(methods):
            #     print(f'{env_id}/{method}')
            #     yKey = "EpRet/mean" if method == 'ppo' else "test/EpRet/mean"
            #     success = loader.read_metrics("time", yKey, path=f"**/{env_id}/{method}/**/metrics.pkl")
            #     # print(success.head())
            #     # config = loader.read_params("env_id")
            #     plot_area(success, "time", yKey, label=method.upper(),
            #               color=COLORS[i % len(COLORS)], x_format="timedelta", y_opts={"scale": "k"})
            #
            # # plt.xlim(0, 1800)
            # plt.xlabel('Wall-clock Time')
            # plt.ylabel('Reward')
            # plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
            # plt.tight_layout()
            # doc.savefig(f"figures/{name}_wall_clock.png", zoom="50%", bbox_inches="tight")
            # plt.close()

    doc.flush()

with doc:
    # launch training
    if not prefix:
        import jaynes
        from firedup.algos.dqn.dqn_her import dqn
        from pg_experiments import instr

        debug = "pydevd" in sys.modules
        jaynes.config("local" if debug else 'cpu-mars')

        for env_id, env_name in zip(env_ids, short_names):
            for seed in [100, 200, 300, 400, 500]:
                thunk = instr(dqn,
                              env_id=env_id,
                              env_kwargs=dict(r=0.04),
                              obs_keys=("x", "goal"),
                              replay_size=40_000,
                              her_k=1,
                              optim_epochs=1,
                              max_ep_len=50,
                              ac_kwargs=dict(hidden_sizes=[32, ] * 2),
                              gamma=0.985,
                              target_update_interval=1000,
                              seed=seed,
                              steps_per_epoch=4000,
                              epsilon_train=0.2,
                              epsilon_decay_period=200_000,
                              epochs=100,
                              _job_prefix="debug" if debug else None,
                              _job_postfix=f"{env_name}/s{seed}")
                jaynes.run(thunk)

        doc.print('Launched@', logger.prefix)

doc @ f"""
Launched@[[{logger.prefix}]](http://localhost:3001{logger.prefix})
"""
