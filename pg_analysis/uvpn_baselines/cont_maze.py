from cmx import doc, CommonMark

# doc = CommonMark(filename="README.md", overwrite=True)
from env_wrappers.flat_env import FlatEnv

doc @ """
# Continuous Control Baselines with Maze Environments

Include ppo, sac, td3 and ddpg.

- [ ] add goal-conditioning
- [ ] add her
"""
with doc:
    # methods = ['ppo', 'sac', 'td3', 'ddpg']
    methods = ['sac']
    env_ids = [
        "ge_world:Maze-v0",
        "ge_world:CMaze-v0",
        "ge_world:HMaze-v0",
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

with doc:
    if not prefix:
        import jaynes
        from firedup.algos.ppo.ppo import ppo
        from firedup.algos.sac.sac import sac
        from firedup.algos.td3.td3 import td3
        from firedup.algos.ddpg.ddpg import ddpg
        from pg_experiments import instr

        jaynes.config()

        for method in methods:
            for env_id, name in zip(env_ids, short_names):
                for seed in [100, 200, 300, 400, 500]:
                    thunk = instr(eval(method),
                                  env_id=env_id,
                                  wrappers=(FlatEnv,),
                                  ac_kwargs=dict(hidden_sizes=[128, ] * 3),
                                  gamma=0.99,
                                  seed=seed,
                                  steps_per_epoch=4000,
                                  epochs=500 if method == "ppo" else 50,
                                  _job_postfix=method)

                    jaynes.run(thunk)

        jaynes.listen()
