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
    prefix = "/geyang/playground/2020/08-15/uvpn_baselines/dqn_maze/17.12.25"

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
    for env_id, name in zip(env_ids, short_names):
        with doc.row():
            plt.figure(figsize=(4.5, 2.8))
            plt.title(name)
            xKey = "__timestamp"
            yKey = "test/success/mean"
            success = loader.read_metrics(xKey, yKey, path=f"**/{name}/**/metrics.pkl")
            plot_area(success, xKey, yKey, label=method.upper(),
                      color=COLORS[i % len(COLORS)], x_format="timedelta", y_opts={"scale": "%"})

            plt.xlabel('Wall-clock Time')
            plt.ylabel('Success')
            plt.ylim(0, 1)
            # plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            doc.savefig(f"figures/dqn_maze/{name}_success.png", zoom="50%", bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(4.5, 2.8))
            plt.title(name)
            xKey = "__timestamp"
            yKey = "test/dist/mean"
            success = loader.read_metrics(xKey, yKey, path=f"**/{name}/**/metrics.pkl")
            plot_area(success, xKey, yKey, label=method.upper(),
                      color=COLORS[i % len(COLORS)], x_format="timedelta", y_opts={"scale": "cm"})

            plt.xlabel('Wall-clock Time')
            plt.ylabel('Distance to Goal')
            # plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            doc.savefig(f"figures/dqn_maze/{name}_dist.png", zoom="50%", bbox_inches='tight')
            plt.close()

    doc.flush()

doc @ """
Launch Script: 
"""
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
                video_interval = 5 if seed == 100 else None
                charts = [dict(type="video", glob="**/*.mp4")] if seed == 100 else []
                thunk = instr(dqn,
                              env_id=env_id,
                              env_kwargs=dict(r=0.02),
                              obs_keys=("x", "goal"),
                              replay_size=40_000,
                              her_k=1,
                              optim_epochs=1,
                              ep_limit=50,
                              ac_kwargs=dict(hidden_sizes=[32, ] * 2),
                              gamma=0.985,
                              target_update_interval=1000,
                              seed=seed,
                              steps_per_epoch=4000,
                              epsilon_train=0.2,
                              epsilon_decay_period=200_000,
                              epochs=70,
                              video_interval=video_interval,
                              _config=dict(charts=["dist/mean", "success/mean", "EpRet/mean", *charts]),
                              _job_prefix="debug" if debug else None,
                              _job_postfix=f"{env_name}/s{seed}")
                jaynes.run(thunk)

        doc.print('Launching@', logger.prefix)
