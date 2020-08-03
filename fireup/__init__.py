# Algorithms
import os
from . import mpi
from playground.algos.ddpg.ddpg import ddpg
from playground.algos.ppo.ppo import ppo
from playground.algos.sac.sac import sac
from playground.algos.td3.td3 import td3
from playground.algos.trpo.trpo import trpo
from playground.algos.vpg.vpg import vpg
from playground.algos.dqn.dqn import dqn

# Loggers
from playground.utils.logx import Logger, EpochLogger

print(__file__[:-11])
with open(os.path.join(__file__[:-11], "VERSION"), 'r') as f:
    __version__ = f.read()
