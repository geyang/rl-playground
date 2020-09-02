
# Metaworld Baselines

Include ppo, sac, td3 and ddpg.

- [ ] add goal-conditioning
- [ ] add her


```python
# methods = ['ppo', 'sac', 'td3', 'ddpg']
# methods = ['sac', 'td3', 'ddpg']
methods = ['sac']
env_prefix = "env_wrappers.metaworld"
env_ids = [
    # f"{env_prefix}:Reach-v1",
    # f"{env_prefix}:Push-v1",
    f"{env_prefix}:Pick-place-v1",
    # f"{env_prefix}:Box-close-v1",
    # f"{env_prefix}:Bin-picking-v1",
]
short_names = [d.split(':')[-1] for d in env_ids]
epochses = [
    # 40,  # 100,
    500,  # 500, 500,
]
ep_limits = [
    # 150,  # 150,
    150,  # 200, 150,
]
prefix = None
```


```python
if not prefix:
    import jaynes
    from ml_logger import logger
    from firedup.algos.ppo.ppo import ppo
    from firedup.algos.sac.sac import sac
    from firedup.algos.td3.td3 import td3
    from firedup.algos.ddpg.ddpg import ddpg
    from pg_experiments import instr

    jaynes.config("local" if "pydevd" in sys.modules else "cpu-mars")

    for method in methods:
        for ent in [0.01, 0.05, 0.2]:
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
                                  replay_size=20_000,
                                  batch_size=256,
                                  lr=1e-3,
                                  alpha=ent,
                                  optimize_alpha=False,
                                  start_steps=10_000,
                                  steps_per_epoch=4000,
                                  epochs=epochs,
                                  video_interval=video_interval,
                                  _config=dict(charts=["success/mean", "reachDist/mean", "goalDist/mean",
                                                       dict(type="video", glob="**/*.mp4")]),
                                  _job_postfix=f"{name}/{method}-ent{ent}")
                    logger.log_text(f"""
                    # On-policy, small buffer

                    Entropy is {ent} in this run
                    """, "README.md", dedent=True)

                    jaynes.run(thunk)

    jaynes.listen()
```


```python
if not prefix:
    import jaynes
    from firedup.algos.ppo.ppo import ppo
    from firedup.algos.sac.sac import sac
    from firedup.algos.td3.td3 import td3
    from firedup.algos.ddpg.ddpg import ddpg
    from pg_experiments import instr

    jaynes.config("local" if "pydevd" in sys.modules else "cpu-mars")

    for method in methods:
        for ent in [0.2, 0.5, 1.0, 2.0]:
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
                                  batch_size=128,
                                  lr=3e-4,
                                  start_steps=1500,
                                  steps_per_epoch=4000,
                                  optimize_alpha=False,
                                  alpha=ent,
                                  epochs=epochs,
                                  video_interval=video_interval,
                                  _config=dict(charts=["success/mean", "reachDist/mean", "goalDist/mean",
                                                       dict(type="video", glob="**/*.mp4")]),
                                  _job_postfix=f"{name}/{method}-ent{ent}")

                    jaynes.run(thunk)

    jaynes.listen()
```

