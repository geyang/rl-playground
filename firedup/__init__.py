# Algorithms
import os
from . import mpi
from firedup.algos.ddpg.ddpg import ddpg
from firedup.algos.ppo.ppo import ppo
from firedup.algos.sac.sac import sac
from firedup.algos.td3.td3 import td3
from firedup.algos.trpo.trpo import trpo
from firedup.algos.vpg.vpg import vpg
from firedup.algos.dqn.dqn import dqn

# Loggers
from firedup.utils.logx import Logger, EpochLogger

print(__file__[:-11])
with open(os.path.join(__file__[:-11], "VERSION"), 'r') as f:
    __version__ = f.read()
