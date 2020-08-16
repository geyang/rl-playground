
# Instantiate A Single Environment

We pick two environments:

```python
tasks = ["box-close-v1", "bin-picking-v1"]
```
```python
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
```
<div style="flex-wrap:nowrap; display:flex; flex-direction:row; item-align:center;"><div><div style="text-align: center">box-close-v1</div><img style="margin:0.5em;" src="videos/box-close-v1_49.gif" /></div><div><div style="text-align: center">bin-picking-v1</div><img style="margin:0.5em;" src="videos/bin-picking-v1_49.gif" /></div></div>
