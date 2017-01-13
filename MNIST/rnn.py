from __future__ import print_function

import tensorflow as tf
import numpy as np


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
        self.num_epochs = 50
        self.num_classes = 10
        self.input_size = 784
        self.input_weight, self.input_height = 28, 28
        self.batch_per_epoch = int(self.mnist.train.num_examples/self.batch_size)
        self.display_step = 1

        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        self.Y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes



    def inference(self):
        """
        RNN
        :return:
        """
        # Reshaping for sequene modelling
        X_seq = tf.reshape(self.X, [-1, 28, 28]) #[-1, 28, 28]

        # Define a lstm cell with tensorflow
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(100)

        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(cell= rnn_cell,
                                    inputs=X_seq,
                                    dtype=tf.float32)

        # Set model weights
        self.W = tf.Variable(tf.zeros([100, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        # Classif using rnn inner loop last output
        self.logits = tf.add(tf.matmul(outputs[:,-1,:], self.W), self.b) # WX+b : the logits
        self.Y_hat = tf.nn.softmax(self.logits) # softmax

    def losses(self):
        """
        Compute the cross entropy loss
        :return:
        """
        # cross entropy loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.Y))



    def optimizer(self):
        """
        Create a optimizer and therefore a training operation
        :return:
        """
        # The optimizer
        self.opt = tf.train.AdagradOptimizer(self.learning_rate)

        # Training operation to run later
        self.train_op = self.opt.minimize(self.loss)


    def metrics(self):
        """
        Compute the accuracy
        :return:
        """
        # Label prediction of the model (the highest one)
        self.predicted_label = tf.argmax(self.Y_hat, 1)
        # Real class:
        self.real_label = tf.argmax(self.Y, 1)
        # Number of correct prediction
        self.correct_prediction = tf.equal(self.predicted_label, self.real_label)
        # Calculate accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy = tf.mul(100.0, self.accuracy)


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
            for epoch in range(self.num_epochs): # 1 epoch = 1 loop over the entire training set
                for s in range(self.batch_per_epoch):
                    # Get bacth fro MNIST training set
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    # Apply the training op
                    (_,
                     loss_train,
                     accuracy_train) = sess.run([self.train_op,
                                                 self.loss,
                                                 self.accuracy],
                                                feed_dict={self.X: batch_xs,
                                                           self.Y: batch_ys})

                    # Print loss and accuracy on the batch
                    if s % 200 == 0:
                        print("\033[1;37;40mStep: %04d , "
                          "TRAIN: loss = %.4f - accuracy = %.2f"
                          % ((epoch*self.batch_per_epoch + s),
                             loss_train, accuracy_train ) )




                # Display logs per epoch step
                if (epoch) % self.display_step == 0:
                    # Compute loss on test (only 200 random images)
                    (loss_test,
                     accuracy_test) = sess.run([self.loss,
                                                self.accuracy],
                                               feed_dict={self.X: mnist.test.images[:100],
                                                          self.Y: mnist.test.labels[:100]})
                    # Compute loss on training (only 200 random images)
                    (loss_train,
                     accuracy_train) = sess.run([self.loss,
                                                 self.accuracy],
                                                feed_dict={self.X: mnist.train.images[:100],
                                                           self.Y: mnist.train.labels[:100]})
                    print("\033[1;32;40mEpoch: %04d , "
                          "TRAIN: loss = %.4f - accuracy = %.2f | "
                          "TEST: loss = %.4f - accuracy = %.2f"
                          % (epoch + 1,
                             loss_train, accuracy_train,
                             loss_test, accuracy_test))


def main(_):
    """
    Main function
    :param _:
    :return:
    """

    # Instanciate a MNIST class
    model = MNIST_logistic(learning_rate=0.01,
                           batch_size=64)
    # Setup the graph
    model.inference()

    # Compute loss and metrics
    model.losses()
    model.metrics()

    # Create an optimzer
    model.optimizer()

    # And finally train your model!
    model.train()



# To start the app for tensorflow
if __name__ == '__main__':
    tf.app.run()

