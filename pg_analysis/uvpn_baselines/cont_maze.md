
# Continuous Control Baselines with Maze Environments

Include ppo, sac, td3 and ddpg.

- [ ] add goal-conditioning
- [ ] add her

``` python
# methods = ['ppo', 'sac', 'td3', 'ddpg']
methods = ['sac']
env_ids = [
    "ge_world:Maze-v0",
    "ge_world:CMaze-v0",
    "ge_world:HMaze-v0",
]
short_names = [d.split(':')[-1] for d in env_ids]
prefix = None
```
