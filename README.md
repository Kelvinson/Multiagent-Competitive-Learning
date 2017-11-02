# Multiagent-Competitive-Learning
try to implement the training phase code for two agents in the competitive environment and  it's imcomplete.

I imported the competitive environment from OpenAI's [repo](https://github.com/openai/multiagent-competition) and revised the ppo algorithm from the [baseline](https://github.com/openai/baselines/tree/master/baselines/ppo1) to adapt to the competitive environment.
As you can see, the dependencies are the dependencies of the two repo above. [baseline](https://github.com/openai/baselines/) etc have to be installed at first.

My idea is simple, in the  compete_learn function in the train_run.py file, I use lists to store the states and rewards, etc  info of the two agents and alternating learning bwtween the two agents. It'a a naive idea, advice are wanted. Thank you!
