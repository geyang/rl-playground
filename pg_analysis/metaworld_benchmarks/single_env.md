
# Instantiate A Single Environment

We pick two environments:


```python
tasks = ["box-close-v1", "bin-picking-v1"]
```


# To-dos

Next step we need to figure out a way to register these environments
with gym, to randomly initialize the tasks.
Need to figure out a way to hook this up to regular gym.

1. register the environment directly
2. set task (figure out where the task comes from, because it is an object)

Then everything is just regular stuff


```python
envs = {}
for task_name in tasks:
    mt1 = metaworld.MT1(task_name)  # Construct the benchmark, sampling tasks

    Env = mt1.train_classes[task_name]
    test_env_classes[task_name] = Env
    envs[task_name] = Env.__name__

doc("The classes are located at:")
doc.yaml(envs)
```

