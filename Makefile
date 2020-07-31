python=/Users/ge/opt/anaconda3/envs/plan2vec/bin/python

help:
	${python} -m playground.run -h
train:
	${python} -m playground.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
evaluate:
	${python} -m playground.run test_policy data/installtest/installtest_s0
plot:
	${python} -m playground.run plot data/installtest/installtest_s0