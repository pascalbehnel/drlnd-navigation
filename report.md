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
Parts of the in this project contained code were taken from a previous exercise, where parts of the code were already implemented, so not everything in this project was coded by myself. 

The Agent class tries to implement the Agent that interacts with the Banana environment. In order to implement the Fixed Targets method, it contains two qnetworks, the local qnetwork, and the target qnetwork. It also contains a reference to a ReplayBuffer object which tries to implement the experience replay method. 

The step function makes sure that we, for one, add the current step information to our memory, and secondly, to trigger the learning step, but only every x steps, which in this case is four.

The learn function - which is called from within the step function - is responsible for the learning step of this algorithm. Thus, it computes the Q_targets as well as Q_expected values, and then uses those to compute the loss, doing backward propagation and updating the weights. It uses the mean-squared-error loss function to compute the loss. As a last step it calls the soft_update function.

Important to notice is that the learn function also implements the Double DQN method of optimizing the DQN Algorithm. For that method, we adapt finding the Q_target values by first choosing the best action based on the local network (the one that gets trained), and then evaluating that action based on the target network (the one that learns at a slower rate). This helps in not over-estimating values especially in the early stages of learning, since the network has not yet adapted and thus, it might over-value some actions.

The soft_update function is used to realize the Fixed Targets method of optimizing the algorithm. It updates the weights of the target network, but it only does it at a rate of TAU, which is set at the top of the dqn_agent.py file. This means it is adapted based on the local model, but it only takes in parts of it and thus becomes decoupled from the local model.

The act method takes in the state and a value for epsilon, and then tries to derive a new action based on the state that the method gets. Epsilon is used to select, if a random action should be chosen, or if the action should be based on our local qnetwork. This mechanic tries to solve the explore-exploit-dilemma.

The ReplayBuffer class is a simple memory that stores state, action, reward, next_state, done pairs, and enables to provide a random sample of these. 

#### Model

The in this implementation used model to approximate the Q-Function is a simple deep neural network. Here is it's shape:

```python
QNetwork(
  (fc1): Linear(in_features=37, out_features=16, bias=True)
  (fc2): Linear(in_features=16, out_features=32, bias=True)
  (fc3): Linear(in_features=32, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=128, bias=True)
  (fc5): Linear(in_features=128, out_features=4, bias=True)
)
```

At first i started off with three layers, but found that three layers turned out a bit better. I also tried starting off with a high number of out_features (f.e. 128) and then getting lower until they reach the action count of 4, but i found that letting the out_features get bigger and bigger after they collapse into the 4 possible actions worked better. I think this might be because, like this, the last layer has a vast amount of features to derive the last 4 actions from, versus the other method where there are more features in the beginning and less at the end.

The forward function is pretty standard and looks like this:

```python
def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x
```

I am using a normal relu for an activation function.

### Plot of Rewards

![](https://github.com/pascalbehnel/drlnd-navigation/blob/main/score-graph.PNG)


### Ideas for Future Work

This implementation could still be optimized by adding Prioritized Experience Replay, which would not just learn from random experiences in the Buffer, but instead would choose experiences it could theoretically learn more from based on the error that was computed for that experience. 

Another way of optimizing could be implementing Dueling DQN, which would split up the linear layer into two parts, one part computing the state values, the other computing the advantage values.

Having all of these implemented would result in a so-called Rainbow implementation of the DQN algorithm.