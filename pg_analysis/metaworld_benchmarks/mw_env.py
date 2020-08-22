from cmx import doc

doc @ """
# gym-metaworld wrapper

We pick two environments:
"""
with doc:
    tasks = ["box-close-v1", "bin-picking-v1"]

with doc, doc.row() as row:
    import metaworld
    from env_wrappers.metaworld import RenderEnv

    for task_name in tasks:
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

        row.video(frames, f"videos/{task_name}.gif", caption=task_name, width=240, height=160)
        env.close()

with doc @ "Now show the reward distribution":
    import matplotlib.pyplot as plt

    plt.hist(rewards, bins=10, histtype='stepfilled')
    doc.savefig(f"figures/single_env/reward_dist.png", dpi=120, zoom="30%")

print('I am finished')
