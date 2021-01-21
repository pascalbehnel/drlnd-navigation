# Report



### Learning Algorithm
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
But just swapping the Q-Table with a neural network wouldn't necessaraly result in the Agent actually converging to an at least somewhat optimal policy. There are multiple tricks that need to be introduced in order to help the agent learn more effectively. In this project, these were as follows:

##### Experience Replay
This technique uses a memory of past status, action, reward, next state, next action pairs, and selects some of these at random and relearns the experience. 

##### Fixed Targets
This technique tries to loosen the bond between 

### Plot of Rewards


### Ideas for Future Work
