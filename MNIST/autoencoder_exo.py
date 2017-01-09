from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/sid/MNIST", one_hot=True)
#mnist = input_data.read_data_sets("/Users/fabien/Datasets/MNIST", one_hot=True)


class Autoencoder(object):
    """
    Class to construct a simple logistic regression on MNIST (i.e a neural net w/o hidden layer)
    """

    def __init__(self, learning_rate, batch_size):
        """
        Init the class with some parameters
        :param learning_rate:
        :param batch_size:
        """
        # Parameters
        self.learning_rate = learning_rate
        self.mnist = mnist
        self.batch_size = batch_size
        self.num_epochs = 5
        self.num_classes = 10
        self.input_size = 784
        self.input_weight, self.input_height = 28, 28
        self.batch_per_epoch = int(self.mnist.train.num_examples / self.batch_size)
        self.display_step = 1

        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784

    def inference(self):
        """
        Design the inference model (here a simple neuralnet)
        :return:
        """

        # Building the encoder
        def encoder(x):
            ?????

        # Building the decoder
        def decoder(x):
            ?????

        # Construct model
        encoder_op = encoder(self.X)
        self.X_reconstruct = decoder(encoder_op)

    def losses(self):
        """
        Compute mean square loss
        :return:
        """
        # cross entropy loss
        self.loss = ?????

    def optimizer(self):
        """
        Create a optimizer and therefore a training operation
        :return:
        """
        # The optimizer
        self.opt = tf.train.AdamOptimizer(self.learning_rate)

        # Training operation to run later
        self.train_op = self.opt.minimize(self.loss)

    def train(self):
        """
        Train the model on MNIST training set
        :return:
        """
        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.num_epochs):  # 1 epoch = 1 loop over the entire training set
                for s in range(self.batch_per_epoch):
                    # Get bacth fro MNIST training set
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    # Apply the training op
                    (_,
                     loss_train) = sess.run([self.train_op,
                                             self.loss],
                                            feed_dict={self.X: batch_xs})
                    # Print loss and accuracy on the batch
                    if s % 200 == 0:
                        print("\033[1;37;40mStep: %04d , "
                              "TRAIN: loss = %.4f"
                              % ((epoch * self.mnist.train.num_examples + s),
                                 loss_train))

                # Display logs per epoch step
                if (epoch) % self.display_step == 0:
                    # Compute loss on validation (only 200 random images)
                    loss_val = sess.run(self.loss,
                                        feed_dict={self.X: mnist.test.images[:200]})

                    # Compute loss on train (only 200 random images)
                    loss_train = sess.run(self.loss,
                                          feed_dict={self.X: mnist.train.images[:200]})

                    print("\033[1;32;40mEpoch: %04d , "
                          "TRAIN: loss = %.4f| "
                          "VALIDATION: loss = %.4f"
                          % (epoch + 1,
                             loss_train,
                             loss_val))

            # Plot reconstruted images
            X_reconstr = sess.run(self.X_reconstruct,
                                  feed_dict={self.X: mnist.test.images[:10]})

            # Compare original images with their reconstructions
            f, a = plt.subplots(2, 10, figsize=(10, 2))
            for i in range(10):
                a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
                a[1][i].imshow(np.reshape(X_reconstr[i], (28, 28)))
            f.show()
            plt.draw()
            plt.waitforbuttonpress()


def main(_):
    """
    Main function
    :param _:
    :return:
    """

    # Instanciate a MNIST class
    model = Autoencoder(learning_rate=0.01,
                        batch_size=100)
    # Setup the graph
    model.inference()

    # Compute loss and metrics
    model.losses()

    # Create an optimzer
    model.optimizer()

    # And finally train your model!
    model.train()


# To start the app for tensorflow
if __name__ == '__main__':
    tf.app.run()

