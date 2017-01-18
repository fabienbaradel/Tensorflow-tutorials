"""
Re-written code from https://github.com/awjuliani/DeepRL-Agents
I only had few omments and a testing part at the end.
"""

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim



env = gym.make('FrozenLake-v0')

np.set_printoptions(suppress=True)

# Q-network: Q(state) = action where Q is a simple matrix
input_state = tf.placeholder(shape=[1, 16], dtype=tf.float32, name="input_state")
W = ?????
Q_values = ?????
action = tf.argmax(Q_values, 1)

# Computing the loss function: sum of squares between Q_target and Q_values
Q_target_values = tf.placeholder(shape=[1, 4], dtype=tf.float32, name="Q_target_values")
diff_tf = ?????
loss = ????


# Define an optimizer to minimize the loss function
opt = ?????
train_op = ?????

# Initialize all the variables of the graph
init = tf.global_variables_initializer()

# Set learning parameters
gamma = .99 # for the Q_target
e = 0.1 # chance of taking a random action at any step
num_episodes = 2000 # total number of episodes during the learning


# Create lists to contain total rewards and steps per episode
jList = []
rList = []

with tf.Session() as sess:
    sess.run(init)

    ############
    # Training #
    ############
    print("\n###########################")
    print("## TRAINING YOUR AGENT ####")
    print("###########################")

    for i in range(num_episodes):
        # Reset environment and get first observation
        s = env.reset()
        R = 0 # sum of the reward over an episode (1.0 if we reach the goal state, 0 otherwise
        j = 0 # number of step in the environment

        # The Q-Network #
        # The episode is stopped if we reach a hole, the goal state or after 100 steps
        while j < 99 and s not in [5,7,11,12,15]:
            j+=1

            # Choose an action by greedily (with e chance of random action) from the Q-network
            s_one_hot = np.identity(16)[s:s + 1] # one hot encoding of the state
            a, all_Q = sess.run([action, Q_values], feed_dict={input_state: ?????})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            # Get new state and reward from environment
            new_s, r, _, _ = env.step(a[0])

            # Obtain the Q values of the new state by feeding the new state through our network still with the same parameters as for the previous step
            new_s_one_hot = np.identity(16)[new_s:new_s + 1]
            all_next_Q = sess.run(Q_values, feed_dict={input_state: ??????})

            # Obtain max(all_next_Q) and set our Q_target_values for chosen action.
            # Here is a little trick for the backpropagation: do not touch this part for the Q value target
            max_all_next_Q = np.max(all_next_Q)
            all_Q_target = all_Q.copy()
            all_Q_target[0, a[0]] = r + gamma * max_all_next_Q # Q_target definition

            # Train our network by minimizing SE(all_Q_target - all_Q)
            sess.run(train_op, feed_dict={input_state: ?????,
                                          Q_target_values: ?????})


            # Update the reward and the current state for the next step
            R += r
            s = new_s

        # Append final reward value and number of step did during the episode
        jList.append(j)
        rList.append(R)

        # Print in the log every 100 episodes
        if i % 100 == 0 and i != 0:
            print("Episodes %04d to %04d: nb of successful episodes = %i/100, nb of steps in avg = %.2f " % (i-100,
                                                                               i,
                                                                               int(sum(rList[-100:])),
                                                                               float(np.mean(jList[-100:]))
                                                                               ))
    # print the W matrix
    W_matrix = sess.run(W)
    print("\nHere is your the matrix estimated by your network:")
    print(("      Left  -    Down    -    Right   -    Up"))
    print(np.asarray(W_matrix))

    ############
    # Testing  #
    ############
    print("\n###########################")
    print("## TESTING YOUR AGENT #####")
    print("###########################")
    # Do 100 episodes with the network whithout random choice to now how good is your agent
    nb_successful_episodes = 0
    total_nb_of_steps = 0

    for i in range(100):
        # Reset environment and get first observation
        s = env.reset()
        R = 0  # sum of the reward over an episode (1.0 if we reach the goal state, 0 otherwise
        j = 0  # number of step in the environment

        while j < 99 and s not in [5,7,11,12,15]:
            j+=1

            # Choose an action from the trained Q-network
            s_one_hot = np.identity(16)[s:s + 1] # one hot encoding of the state
            a, all_Q = sess.run([action, Q_values], feed_dict={input_state: ?????})

            # Get new state and reward from environment
            new_s, r, _, _ = env.step(a[0])

            # Update the reward and the current state for the next step
            R += r
            s = new_s

        # Add the final reward after the episode and nb of steps
        nb_successful_episodes += R
        total_nb_of_steps +=j

    print("Testing on 100 episodes: \nYour agent has finished %i episodes and did %.2f steps in average!" % (int(nb_successful_episodes),
                                                                                                       total_nb_of_steps/100.0))
    if nb_successful_episodes < 30:
        print("=> Keep coding or maybe just re-run your script...")
    else:
        print("=> Well done! How to improve your score and make your agent smarter?!\n")

