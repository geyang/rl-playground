import sys
from cmx import doc

from env_wrappers.flat_env import FlatEnv
from env_wrappers.metaworld import ALL_ENVS
from firedup.wrappers import singleton_env_fn

doc @ """
# Metaworld Baselines

Include ppo, sac, td3 and ddpg.

- [ ] add goal-conditioning
- [ ] add her
"""
with doc:
    # methods = ['ppo', 'sac', 'td3', 'ddpg']
    # methods = ['sac', 'td3', 'ddpg']
    methods = ['sac']
    env_prefix = "env_wrappers.metaworld"
    env_ids = [
        f"{env_prefix}:Reach-v1",
        f"{env_prefix}:Push-v1",
        f"{env_prefix}:Drawer-open-v1",
        f"{env_prefix}:Button-press-down-v1",
        f"{env_prefix}:Pick-place-v1",
        f"{env_prefix}:Box-close-v1",
        f"{env_prefix}:Bin-picking-v1",
    ]
    short_names = [d.split(':')[-1] for d in env_ids]
    epochses = [
        40, 100,
        500, 500, 500,
    ]
    ep_limits = [
        150, 150,
        150, 200, 150,
    ]
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
    for env_id, name in zip(env_ids, short_names):
        with doc.row():
            # plt.figure(figsize=(4.5, 2.8))
            plt.title(name)
            xKey = "__timestamp"
            yKey = "test/success/mean"
            for i, method in enumerate(methods):
                success = loader.read_metrics(xKey, yKey, path=f"**/{name}/**/metrics.pkl")
                plot_area(success, xKey, yKey, label=method.upper(),
                          color=COLORS[i % len(COLORS)], x_format="timedelta", y_opts={"scale": "%"})

            plt.xlabel('Wall-clock Time')
            plt.ylabel('Success')
            plt.ylim(0, 1)
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            doc.savefig(f"figures/cont_maze/{name}_success.png", zoom="50%", bbox_inches='tight')
            plt.close()

            # plt.figure(figsize=(4.5, 2.8))
            plt.title(name)
            xKey = "__timestamp"
            yKey = "test/dist/mean"
            for i, method in enumerate(methods):
                success = loader.read_metrics(xKey, yKey, path=f"**/{name}/**/metrics.pkl")
                plot_area(success, xKey, yKey, label=method.upper(),
                          color=COLORS[i % len(COLORS)], x_format="timedelta", y_opts={"scale": "cm"})

            plt.xlabel('Wall-clock Time')
            plt.ylabel('Distance to Goal')
            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            doc.savefig(f"figures/cont_maze/{name}_dist.png", zoom="50%", bbox_inches='tight')
            plt.close()

    doc.flush()

with doc:
    if not prefix:
        import jaynes
        from ml_logger import logger
        from firedup.algos.sac_2.sac import sac
        from pg_experiments import instr

        jaynes.config("local" if "pydevd" in sys.modules else "cpu-mars")

        for method in methods:
            # for ent in [0.01, 0.05, 0.2]:
            for env_id, name, epochs, ep_limit in zip(env_ids, short_names, epochses, ep_limits):
                for seed in [100, 200, 300]:
                    # video_interval = 1 if seed == 100 else None
                    video_interval = 5
                    thunk = instr(eval(method),
                                  env_id=env_id,
                                  seed=seed,
                                  # env_fn=singleton_env_fn,
                                  ac_kwargs=dict(hidden_sizes=[400, ] * 3),
                                  gamma=0.99,
                                  # standard for metaworld
                                  ep_limit=ep_limit,
                                  replay_size=1000_000,
                                  batch_size=256,
                                  lr=1e-3,
                                  # alpha=ent,
                                  # optimize_alpha=False,
                                  # start_steps=10_000,
                                  steps_per_epoch=4000,
                                  epochs=epochs,
                                  video_interval=video_interval,
                                  _config=dict(charts=["success/mean", "reachDist/mean", "goalDist/mean",
                                                       dict(type="video", glob="**/*.mp4")]),
                                  _job_postfix=f"{name}/{method}")
                    logger.log_text(f"""
                    # New SAC implementation from spinning up

                    Notable change: there is no entropy regularization
                    """, "README.md", dedent=True)

                    jaynes.run(thunk)

        jaynes.listen()
