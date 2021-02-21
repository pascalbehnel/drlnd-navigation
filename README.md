# Description

This Project is about training an Agent to play a game, where the Agent needs to traverse a virtual 3D world, where Bananas are scattered all over the place. But not all Bananas are created equal! Some bananas are yellow, those are good bananas, and some are blue, those are bad ones. 

The goal for the agent is to traverse the enivornment, and try to collect as many yellow bananas as possible, while avoiding the blue ones. Yellow bananas yield a reward of +1, while blue ones will result in a penalty of 1, so a reward of -1.

The agent is able to perform 4 actions:
- turn left 45°
- turn right 45°
- move forward
- move backward

The environment is considered solved, if the Agent is able to collect an average of +13 points over the course of 100 episodes.

# Repository Content
- repord.md: A document that tries to describe the Algorithm behind this implementation
- Navigation.ipynb: A jupyter notebook that can be used to train a new agent, or load a pre-trained one, and use it to traverse the environment

# Requirements

To run this implementation, follow these steps:
- Create a new environment

  ```
  conda create --name banana python=3.6
  activate banana
  ```

- install OpenAI gym

  ```
  pip install gym
  ```

- install further dependencies

  ```
  cd python
  pip install -r requirements.txt
  ```

- Download the Unity Environment that is matching your operating system (thanks to Udacity)

  - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

- Start jupyter notebook. Make sure that it is at the root of this project

  ```
  jupyter notebook "<folder>"
  ```
