
# Continuous Control Baselines with Sawyer Robot

Include ppo, sac, td3 and ddpg.

- [ ] add goal-conditioning
- [ ] add her

```python
# methods = ['ppo', 'sac', 'td3', 'ddpg']
# methods = ['sac', 'td3', 'ddpg']
methods = ['sac']
env_ids = [
    "sawyer:Push-v0",
    "sawyer:PickPlace-v0",
    "sawyer:Peg3D-v0",
]
short_names = [d.split(':')[-1] for d in env_ids]
prefix = None
```
