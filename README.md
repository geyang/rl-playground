# Welcome to RL-Playground!

### Installing RL Playground

```
git clone https://github.com/geyang/rl-playground.git
cd rl-playground
pip install -e .
```

RL Playground defaults to installing everything in Gym **except** the MuJoCo environments.

### Check Your Install

To see if you've successfully installed RL Playground, try running PPO in the `LunarLander-v2` environment with:

```
python -m playground.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
```

After it finishes training, watch a video of the trained policy with:

```
python -m playground.run test_policy data/installtest/installtest_s0
```

And plot the results with:

```
python -m playground.run plot data/installtest/installtest_s0
```

## Algorithms

The following algorithms are implemented in the RL Playground package:

- Proximal Policy Optimization (PPO)
- Deep Q-Network (DQN)
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)
- Soft Actor-Critic (SAC)

## Citation

If RL Playground has helped in accelerating the publication of your
paper, please kindly consider citing this repo with the following
bibtext entry:

```bibtex
@misc{yang2020playground,
  author={Ge Yang},
  title={Playground},
  url={https://github.com/geyang/rl_playground},
  year={2019}
}
```
