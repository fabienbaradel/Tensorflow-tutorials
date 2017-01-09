from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.01
training_steps = 1000
display_step = 25

# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]
BATCH_SIZE = n_samples


# Create the placeholder for X and Y
X = tf.placeholder("float", shape=[None])
Y = tf.placeholder("float", shape=[None])

# Set model weights
W = tf.Variable([0.0], name="weight")
b = tf.Variable([0.0], name="bias")

# Construct a linear model: Y_hat = W*X + b
Y_hat = ????????

# Write your loss function: Mean squared error
loss = ????????

# Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit with a mini-batch of data
    for step in range(training_steps):
        # Uniform batch sampling in the training set
        location = np.random.choice(range(n_samples), BATCH_SIZE)
        mini_batch_X, mini_batch_Y = train_X[location], train_Y[location]
        sess.run(train_op, feed_dict={X: mini_batch_X, Y: mini_batch_Y})

        # Display logs per step
        if (step+1) % display_step == 0 or step == 0:
            (loss_value, W_value, b_value) = sess.run([loss, W, b], feed_dict={X: train_X, Y:train_Y})
            print("Step: %04d , Loss = %.4f , W = %.3f , b = %.3f"
                  %(step+1, loss_value, W_value, b_value))
            #plt.plot(train_X, W_value * train_X + b_value)

    print("Optimization Finished!")
    (loss_value_training, W_value, b_value) = sess.run([loss, W, b], feed_dict={X: train_X, Y: train_Y})


    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, W_value * train_X + b_value, label='Fitted line - SGD')
    plt.show()



# closed form solution for W and B
X = np.vstack([train_X, np.ones(len(train_X))]).T
X.shape
Y = train_Y
Y.shape
W_closed_form, b_closed_form  = np.linalg.lstsq(X, Y)[0]

plt.plot(train_X, W_closed_form * train_X + b_closed_form, label='Fitted line - closed form')
plt.legend(loc=0)
plt.show()

print("Closed form: W = %.3f , b = %.3f " % ( W_closed_form, b_closed_form))
print("Gradient descent: W = %.3f , b = %.3f " % (W_value, b_value))
