if __name__ == '__main__':
    from cmx import doc

    doc @ """
    # Instantiate A Single Environment
    
    We pick two environments:
    """
    with doc:
        tasks = ["box-close-v1", "bin-picking-v1"]
    doc @ """
    # To-dos
    
    Next step we need to figure out a way to register these environments
    with gym, to randomly initialize the tasks.
    Need to figure out a way to hook this up to regular gym.

    1. register the environment directly
    2. set task (figure out where the task comes from, because it is an object)

    Then everything is just regular stuff
    """

    import metaworld

    test_env_classes = {}
    with doc:
        envs = {}
        for task_name in tasks:
            mt1 = metaworld.MT1(task_name)  # Construct the benchmark, sampling tasks

            Env = mt1.train_classes[task_name]
            test_env_classes[task_name] = Env
            envs[task_name] = Env.__name__

        doc("The classes are located at:")
        doc.yaml(envs)

    doc @ """
    # Self-Contained Single-Task Environment Example
    
    Below is a self.contained example of how to instantiate a single task environment: 
    """
    with doc, doc.row() as row:
        from env_wrappers.metaworld.mw_env import RenderEnv

        rewards = []
        for task_name, Env in test_env_classes.items():
            mt1 = metaworld.MT1(task_name)  # Construct the benchmark, sampling tasks

            Env = mt1.train_classes[task_name]
            env = Env()  # Create an environment with task `pick_place`
            env = RenderEnv(env)

            frames = []
            for t_id, task in enumerate(mt1.train_tasks):
                env.set_task(task)  # Set task
                obs = env.reset()  # Reset environment
                a = env.action_space.sample()  # Sample an action
                for i in range(5):
                    obs, r, done, info = env.step(a)  # Step the environment with the sampled
                    img = env.render("rgb", width=72, height=48)
                    frames.append(img)
                    rewards.append(r)

            row.video(frames, f"videos/{task_name}.gif", caption=task_name, width=240, height=160)
            env.close()

        from sawyer.misc import obs2dict
        row.yaml(obs2dict(obs))

    with doc @ "Now show the reward distribution":
        import matplotlib.pyplot as plt

        plt.hist(rewards, bins=10, histtype='stepfilled')
        doc.savefig(f"figures/single_env/reward_dist.png", dpi=120, zoom="30%")

    print('I am finished')
