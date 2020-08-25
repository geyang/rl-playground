
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
    f"{env_prefix}:Box-close-v1",
    f"{env_prefix}:Bin-picking-v1",
]
short_names = [d.split(':')[-1] for d in env_ids]
epochses = [40, 100, 2500, 2500, 2500]
ep_limits = [150, 150, 150, 200, 150]
prefix = None
```

