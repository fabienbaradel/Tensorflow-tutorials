import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

# Take an evironnmenet and have a look at it
env = gym.make('FrozenLake-v0')
#env.render()

# See the action space:
print(env.action_space) # 0=Left, 1=Down, 2=Right, 3=Up
# State space = range(16)
tuple_actions = ('Left', 'Down', 'Right', 'Up')


# Random walk in teh frozen lake
def random_walk(env):
    # Reset the environnement
    env.reset()

    # Look at the initial environnement
    print("Initial state")
    env.render()

    # Reward and 'is it done?' of the first state
    reward, done = False, 0.0
    num_iter = 0
    x, y = 0, 0 # initial coordinate

    # Random walk until we win or we fall in a hole
    while done == 0.0:
        num_iter += 1
        # Take a random action and apply it
        a = env.action_space.sample()
        observation, reward, done, info = env.step(a)

        # Print the environment and your new position
        print("\nIter: "+str(num_iter))
        print("Random action: "+str(a)+" = "+tuple_actions[a])
        print("State: "+str(observation))

        env.render()

    # Check the random walk wins
    if reward == 1.0:
        print("\nRandom walk reachs the GOAL after %i iterations!" % num_iter)
    else:
        print("\nYou felt down in a hole after %i iterations..." % num_iter)

random_walk(env)



#These lines establish the feed-forward part of the network used to choose actions

