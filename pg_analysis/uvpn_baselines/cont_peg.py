import gym
import sys

from cmx import doc
from env_wrappers.flat_env import FlatEnv

doc @ """
# Continuous Control Baselines with Maze Environments

Include ppo, sac, td3 and ddpg.

- [ ] add goal-conditioning
- [ ] add her

The default `in_slot` is set to 10% of the time
"""
with doc:
    env_id = "ge_world:Peg2D-v0"
    env = gym.make(env_id, free=False, obs_keys=['x', 'goal'])

doc.yaml({k: f"Shape{v.shape}" for k, v in env.observation_space.spaces.items()})

env.seed(100)
with doc.row() as row:
    for i in range(5):
        obs = env.reset()
        img = env.render("rgb", width=100, height=100)
        row.image(img)

with doc("We show the initial position with `in_slot` set to 0 here"):
    env = gym.make(env_id, free=False, obs_keys=['x', 'goal'], in_slot=None)

env.seed(100)
with doc.row() as row:
    for i in range(5):
        obs = env.reset()
        img = env.render("rgb", width=100, height=100)
        row.image(img)
    print("hey")

with doc @ """Now launch:""":
    # methods = ['ppo', 'sac', 'td3', 'ddpg']
    methods = ['sac', 'td3', 'ddpg']
    env_ids = ["ge_world:Peg2D-reach-v0", "ge_world:Peg2D-v0"]
    test_env_kwargses = [None, dict(in_slot=None)]
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

        jaynes.config("local" if "pydevd" in sys.modules else "cpu-mars")

        for method in methods:
            for env_id, name, tek in zip(env_ids, short_names, test_env_kwargses):
                for seed in [100, 200, 300, 400, 500]:
                    video_interval = 1 if seed == 100 else None
                    charts = [dict(type="video", glob="**/*.mp4")] if seed == 100 else []
                    thunk = instr(eval(method),
                                  env_id=env_id,
                                  seed=seed,
                                  env_kwargs=dict(),
                                  test_env_kwargs=tek,
                                  wrappers=(FlatEnv,),
                                  ac_kwargs=dict(hidden_sizes=[128, ] * 3),
                                  gamma=0.99,
                                  ep_limit=50,
                                  steps_per_epoch=4000,
                                  epochs=500 if method == "ppo" else 10,
                                  video_interval=video_interval,
                                  _config=dict(charts=["success/mean", "dist/mean", *charts]),
                                  _job_postfix=f"{name}/{method}")

                    jaynes.run(thunk)

        jaynes.listen()
