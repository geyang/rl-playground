if __name__ == '__main__':
    from cmx import doc

    doc @ """
    # List of environments from metaworld
    
    We take the simple MT50 (multi-task 10 suite)
    """

    with doc, doc.row():
        import metaworld
        import random
        from env_wrappers.metaworld import RenderEnv

        mt10 = metaworld.MT50()  # Construct the benchmark, sampling tasks
        for name, __Env__ in mt10.train_classes.items():
            print(name)
            env = __Env__()
            task = random.choice([task for task in mt10.train_tasks if task.env_name == name])
            env.set_task(task)
            env = RenderEnv(env)
            env.reset()
            img = env.render('rgb', width=240, height=160)
            doc.image(img, f"figures/task_{name}.png", caption=name)
            env.close()

    doc.flush()
