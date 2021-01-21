# Report



### Learning Algorithm (general concept)
The learning algorithm used in this project is based off of the deep Q-Learning approach as published within the following paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

The deep Q-Learning approach is an enhancement of the Q-Learning algorithm. 

#### Q-Learning
Q-Learning is a reinforcement learning algorithm which is model-free, which means that it doesn't require any model of the environment that the agent wants to act in. Instead, based on the reward that the environment sends back to the agent, and also based on the previous action and state, the algorithm is now able to approximate a so called Q-Table (hence the name Q-Learning), which then can be used to estimate the value of any action, given a specific state of the environment. Utilizing the approximated Q-Table, it is then possible to derive the most optimal action given for any state. 
Q-Learning is proven to arrive at an optimal policy, as long as it has infinite exploration time, and also utilizes a partly random policy. The partly random policy combined with infinite exploration time is important, sothat the agent keeps on exploring the environment and we are sure that at some point the agent has visited every possible state.
The values within the Q-Table get updated with a combination of the old approximation, the learning rate, the discount factor, and the estimate of the future optimal value.

#### Deep Q-Learning
The deep Q-Learning approach in this project is based on this Paper:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

A problem with Q-Learning is that, if we have too many states within our environment, it might be impossible to put each and every state into a Table, since the table would simply explode, and every specific entry wouldn't get updated much, hence there wouldn't be much learning happening.
The idea to solve this issue is, instead of approximating a table, we try to approximate a function, and in this case, the function is a deep neural network, hence the name Deep Q-Learning.
But just swapping the Q-Table with a neural network wouldn't necessarily result in the Agent actually converging to an at least somewhat optimal policy. There are multiple tricks that need to be introduced in order to help the agent learn more effectively. In this project, these were as follows:

##### Experience Replay
This technique can be thought of as a simulation of memory, from which the respective person - or in this case the agent - is learning, after the event actually happened. Of course in reality this is way different compared to how it happens in the computer, but at least the general idea is similar. The computer assembles experiences he makes, and puts them into memory, from where he pulls them out randomly and adapts the Q-Function again. It is important to do this in a random fashion, because in that way, there is no correlation between the learning experience and the previous one.

##### Fixed Targets
A problem with the normal way the update of the Q-Function is set up is, that we adapt our Q-Function by determining the difference between the value of our Q-Function for the old state-action-pair, and the state-action-pair for the new state, and the action that yields the most expected result. Both of these access the Q-Function itself and are thus somewhat coupled together, which isn't helpful if we are trying to find the Q-Function that optimally approximates the expected reward. 
In order to decouple this, while updating our Q-Function, we are still using the same network for both the target and the old value, but for the target, we are actually using a set of weights that only gets updated every x steps, f.e. every 100 steps, but this is a hyper-parameter that we can adapt to our choosing. For all the steps inbetween, the Target weights do not change at all and stay the way they are, and only the old value gets approximated using the most recent weights.
This way, there is less correlation between target and old value, which should help during the learning process.



### Learning Algorithm (implementation)

This project tries to implement the above defined learning algorithm, deep Q-Learning. It also tries to implement experience replay as well as fixed targets, in order to help the q-function to converge towards a successful approximation of the state-value-function. 
Parts of the in this project contained code were taken from a previous exercise, where parts of the code were already implemented, so not everything in this project was coded by myself. Still i will try to explain even the parts that i did not write myself.

#### Model

The in this implementation used model to approximate the Q-Function is a simple deep neural network. 


### Plot of Rewards


### Ideas for Future Work
