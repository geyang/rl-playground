from cmx import doc
import gym
from sawyer.misc import space2dict

doc @ """
# Inspect the Observation Space of gym fetch robots

List of Environments
"""
fetch_envs = [
    "FetchSlide-v1",
    "FetchPickAndPlace-v1",
    "FetchReach-v1",
    "FetchPush-v1"
]

with doc.row() as row:
    for env_id in fetch_envs:
        env = gym.make(env_id)
        obs = env.reset()
        doc(f"""
        ## {env_id}
        spec:
        """)
        doc.yaml(env.spec)
        doc(f"""
        observation space
        """)
        doc.yaml(space2dict(env.observation_space))
        img = env.render('rgb_array', )
        row.image(img, zoom="20%")

doc.flush()
