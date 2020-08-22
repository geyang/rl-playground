
# Continuous Control Baselines with Sawyer Robot

use frame_skip = 4

Include ppo, sac, td3 and ddpg.

- [ ] add goal-conditioning
- [ ] add her

There is some inconsistency between the definition of the observation_space
and the actual observations.


```python
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
```

