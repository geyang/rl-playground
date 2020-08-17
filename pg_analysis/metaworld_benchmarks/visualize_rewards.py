if __name__ == '__main__':
    tasks = ["box-close-v1", "bin-picking-v1"]

    from ml_logger import logger
    rewards = logger.load_pkl("data/rewards.pkl")
    # plt.hist(rewards, bins=10, histtype='stepfilled')
    # logger.savefig(f"figures/single_env/reward_dist.png", dpi=120)

    print('I am finished')
