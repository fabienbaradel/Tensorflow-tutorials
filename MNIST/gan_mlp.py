from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim



# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/sid/MNIST", one_hot=True)
#mnist = input_data.read_data_sets("/Users/fabien/Datasets/MNIST", one_hot=True)




class MNIST_logistic(object):
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
        self.batch_per_epoch = int(self.mnist.train.num_examples/self.batch_size)
        self.display_step = 1

        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        self.Z = tf.placeholder(tf.float32, [None, 100]) # noise for the generator

        # Loss and accuracy tracking
        self.list_loss_train = []
        self.list_loss_validation = []
        self.list_accuracy_train = []
        self.list_accuracy_validation = []



    def inference(self):
        def generator(z):
            net = slim.fully_connected(z, 128)
            generated_image = slim.fully_connected(net, 784, activation_fn=None)
            return generated_image

        def discriminator(x):
            net = slim.fully_connected(x, 128)
            is_real = slim.fully_connected(net, 1, activation_fn=None)
            return is_real

        self.generated_X = generator(self.Z)
        self.D_real = discriminator(self.X)
        self.D_fake = discriminator(self.generated_X)




    def losses(self):
        """
        GAN losses
        :return:
        """
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(self.D_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake, tf.zeros_like(self.D_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake, tf.ones_like(self.D_fake)))


    def optimizer(self):
        """
        Create a optimizer and therefore a training operation
        :return:
        """
        # The optimizer for D and G
        self.opt_D = tf.train.AdamOptimizer(self.learning_rate)
        self.opt_G = tf.train.AdamOptimizer(self.learning_rate)

        # Training operation for D and G solver (minmax problem)
        self.train_op_D = self.opt_D.minimize(self.D_loss)
        self.train_op_G = self.opt_G.minimize(self.G_loss)



    def train(self):
        """
        Train the model on MNIST training set
        :return:
        """
        # Initializing the variables
        init = tf.global_variables_initializer()

        def sample_Z(m, n):
            '''Uniform prior for G(Z)'''
            return np.random.uniform(-1., 1., size=[m, n])

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.num_epochs): # 1 epoch = 1 loop over the entire training set
                # Averaging loss over the epoch
                avg_loss_train = 0.0
                for s in range(self.batch_per_epoch):
                    # Get bacth fro MNIST training set
                    batch_xs, _ = mnist.train.next_batch(self.batch_size)

                    # Apply the training op alternatively
                    (_, loss_D) = sess.run([self.train_op_D, self.D_loss],
                                              feed_dict={self.X: batch_xs,
                                                                             self.Z: sample_Z(self.batch_size, 100)})
                    (_, loss_G) = sess.run([self.train_op_G, self.G_loss],
                                              feed_dict={self.Z: sample_Z(self.batch_size, 100)})

                    # Print loss and accuracy on the batch
                    print(loss_D, loss_G)




def main(_):
    """
    Main function
    :param _:
    :return:
    """

    # Instanciate a MNIST class
    model = MNIST_logistic(learning_rate=0.01,
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

