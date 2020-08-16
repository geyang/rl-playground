if __name__ == '__main__':
    from cmx import doc

    doc @ """
    # Instantiate A Single Environment
    
    We pick two environments:
    """
    with doc:
        tasks = ["box-close-v1", "bin-picking-v1"]

    import metaworld
    from env_wrappers.metaworld import RenderEnv

    with doc, doc.row(wrap=False):
        rewards = []
        for task_name in tasks:
            ml1 = metaworld.ML1(task_name)  # Construct the benchmark, sampling tasks

            frames = []

            env = ml1.train_classes[task_name]()  # Create an environment with task `pick_place`
            env = RenderEnv(env)
            for t_id, task in enumerate(ml1.train_tasks):
                env.set_task(task)  # Set task
                obs = env.reset()  # Reset environment
                a = env.action_space.sample()  # Sample an action
                for i in range(5):
                    obs, reward, done, info = env.step(a)  # Step the environment with the sampled
                    img = env.render("rgb", width=240, height=160)
                    frames.append(img)
                    rewards.append(rewards)

            doc.video(frames, f"videos/{task_name}_{t_id}.gif", caption=task_name)

    with doc @ "Now show the reward distribution", doc.row(wrap=False):
        import matplotlib.pyplot as plt

        plt.hist(rewards, bins=10)
        doc.savefig(f"figures/single_env/reward_dist.png")
