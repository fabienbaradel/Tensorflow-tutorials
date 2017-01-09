from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
BATCH_SIZE = 10
learning_rate = 0.1
training_steps = 100
display_step = 10

# Training Data
training_size = 100
coord_X = np.random.uniform(0, 1, training_size)
coord_Y = np.random.uniform(0, 1, training_size)
color = np.asarray([0 if x+y>1 else 1 for x,y in zip(coord_X,coord_Y)])

coord_XY = np.asarray(zip(coord_X, coord_Y))
n_samples = len(color)


# Create the placeholder for X and Y
X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name = 'X')
Y = tf.placeholder(dtype=tf.float32, shape=[None], name = 'Y')
Y_reshape = tf.reshape(Y, [-1, 1])


# Set model weights
W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

# Construct the inference
Y_hat = ??????????

# Write your loss function: Mean squared error
loss = ??????????

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit the data
    for step in range(training_steps):
        # Uniform batch sampling in the training set
        location = np.random.choice(range(n_samples), BATCH_SIZE)
        mini_batch_X, mini_batch_Y = coord_XY[location], color[location]
        sess.run(train_op, feed_dict={X: mini_batch_X, Y: mini_batch_Y})

        # Display logs per step
        if (step+1) % display_step == 0 or step < 10:
            (loss_value, W_value, b_value) = sess.run([loss, W, b], feed_dict={X: coord_XY, Y:color})
            print("Step: %04d , Loss = %.4f , W-1 = (%.2f, %.2f), b-1 = %.2f"
                  %(step+1, loss_value, W_value[0], W_value[0], b_value))

    print("Optimization Finished!")
    print("")
    (loss_value_training, W_value, b_value) = sess.run([loss, W, b], feed_dict={X: coord_XY, Y: color})


    # Graphic display and fitted line
    # Color for the plot
    plot_color = ['r' if c == 0 else 'g' for c in color]
    plt.scatter(coord_X, coord_Y, color=plot_color, label='Original data')
    xx = np.linspace(0, 1)
    yy = (-W_value[0]/W_value[1]) * np.linspace(0, 1) - (b_value) / W_value[1]
    plt.plot(xx, yy, 'k-', label='Fitted line - SGD')
    plt.legend()
    plt.show()




#from sklearn import linear_model
#logreg = linear_model.LogisticRegression()
#logreg.fit(coord_XY, color)
#logreg.coef_
#logreg.intercept_