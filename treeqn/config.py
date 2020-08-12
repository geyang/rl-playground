import sys
import torch
from params_proto.neo_proto import ParamsProto


class Config(ParamsProto):
    debug = "pydevd" in sys.modules.keys()
    seed = 100
    # config = './conf/default.yaml'
    # config_filename = 'default.yaml'
    #
    # label = 'default'
    # name = 'default'

    description = 'Default atari'
    input_mode = "atari"
    # env_id = 'Seaquest'
    env_id = 'Krull'
    nstack = 4
    frameskip = 10
    use_actor_critic = False
    architecture = 'treeqn'
    tree_depth = 2
    extra_layers = 0
    transition_fun_name = 'two_layer'
    transition_nonlin = 'tanh'
    value_aggregation = 'softmax'
    normalise_state = True
    residual_transition = True
    embedding_dim = 512
    td_lambda = 0.8
    predict_rewards = True
    rew_loss_coef = 1.0
    st_loss_coef = 0.0
    subtree_loss_coef = 0.0
    gamma = 0.99
    million_frames = 400
    eps_million_frames = 4

    n_envs = 15
    nsteps = 5
    target_update_interval = 40000
    alpha = 0.99
    epsilon = 1e-05
    lr = 0.0001
    lrschedule = 'constant'
    max_grad_norm = 5
    vf_coef = 0.5
    ent_coef = 0.01
    # save_folder = './results/temp/'
    log_interval = 1000
    debug_log = True
    number_checkpoints = 3
    log_rolling_window = 250

    device = "cpu"

    @classmethod
    def __init__(cls, _deps=None, **kwargs):
        import os
        from ml_logger import logger

        cls._update(_deps, **kwargs)

        if cls.debug:
            cls.n_envs = 1
            cls.nsteps = 1000  # so that we can collect the rewards
            cls.log_interval = 100

        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.monitor_dir = os.environ.get("$TMPDIR") + f"/monitor-{logger.now: %Y-%d-%m-%f}"
