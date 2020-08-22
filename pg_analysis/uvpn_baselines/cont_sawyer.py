import sys
from cmx import doc

from env_wrappers.flat_env import FlatEnv, FlatGoalEnv

doc @ """
# Continuous Control Baselines with Sawyer Robot

use frame_skip = 4

Include ppo, sac, td3 and ddpg.

- [ ] add goal-conditioning
- [ ] add her

There is some inconsistency between the definition of the observation_space
and the actual observations.
"""
with doc:
    # methods = ['ppo', 'sac', 'td3', 'ddpg']
    methods = ['sac', 'td3', 'ddpg']
    # methods = ['sac']
    env_ids = [
        "sawyer:Reach-v0",
        "sawyer:Peg3D-v0",
        "sawyer:Push-v0",
        "sawyer:PushMove-v0",
        "sawyer:PickPlace-v0",
    ]
    test_kwargses = [
        None,
        dict(init_mode="hover"),
        dict(init_mode="hover"),
        dict(init_mode="hover"),
        dict(init_mode="hover"),
    ]
    short_names = [d.split(':')[-1] for d in env_ids]
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
        from firedup.algos.ppo.ppo import ppo
        from firedup.algos.sac.sac import sac
        from firedup.algos.td3.td3 import td3
        from firedup.algos.ddpg.ddpg import ddpg
        from pg_experiments import instr

        is_debug = "pydevd" in sys.modules
        jaynes.config("local" if is_debug else "cpu-mars", launch=dict(timeout=None if is_debug else 0.01))
        # jaynes.config("cpu-mars", launch=dict(timeout=None))

        for method in methods:
            for env_id, name, t_kwargs in zip(env_ids, short_names, test_kwargses):
                for seed in [100, 200, 300]:
                    # video_interval = 1 if seed == 100 else None
                    video_interval = 5
                    thunk = instr(eval(method),
                                  env_id=env_id,
                                  seed=seed,
                                  test_env_kwargs=t_kwargs,
                                  wrappers=(FlatGoalEnv,),
                                  ac_kwargs=dict(hidden_sizes=[20, ] * 4),
                                  gamma=0.985,
                                  ep_limit=50,
                                  steps_per_epoch=4000,
                                  epochs=500,
                                  video_interval=video_interval,
                                  _config=dict(charts=["success/mean", "dist/mean",
                                                       dict(type="video", glob="**/*.mp4")]),
                                  _job_postfix=f"{name}/{method}")

                    jaynes.run(thunk)

        jaynes.listen()
