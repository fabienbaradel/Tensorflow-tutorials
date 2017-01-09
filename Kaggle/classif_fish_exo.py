from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from numpy import random
import csv


class Fish_classif(object):
    """
    Fish classif from Inception_v3 features (2048)
    """

    def __init__(self, learning_rate, batch_size, num_epoch):
        """
        Init the class with some parameters
        :param learning_rate:
        :param batch_size:
        """
        # Parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epoch
        self.num_classes = 8
        self.input_size = 2048
        self.training_size = 3720
        self.batch_per_epoch = int(self.training_size/self.batch_size)
        self.display_step = 1

        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, 2048])
        self.Y = tf.placeholder(tf.float32, [None])

        # one hot encoding
        self.Y_one_hot = tf.one_hot(tf.cast(self.Y, tf.int32), self.num_classes)


        # Load dataset
        print("Dataset loading...")
        self.data_X = np.loadtxt('fish_features.txt')
        self.data_Y = np.loadtxt('fish_labels.txt')
        print(str(self.data_X.shape[0])+" training data.")

        # Split train / validation
        ind = range(self.data_X.shape[0])
        random.shuffle(ind)
        ind_train, ind_val = ind[:self.training_size], ind[self.training_size:]
        self.train_X, self.train_Y = self.data_X[ind_train, :], self.data_Y[ind_train]
        self.val_X, self.val_Y = self.data_X[ind_val, :], self.data_Y[ind_val]

        # Test
        self.test_X = np.loadtxt('fish_features_test.txt')

        print("Done!")



    def inference(self):
        """
        Softmax regression
        :return:
        """

        # Construct the inference
        self.logits = ???????
        self.Y_hat = ???????

    def losses(self):
        """
        Compute the cross entropy loss
        :return:
        """
        # cross entropy loss
        self.loss = ??????



    def optimizer(self):
        """
        Create a optimizer and therefore a training operation
        :return:
        """
        ???????


    def metrics(self):
        """
        Compute the accuracy
        :return:
        """
        # Label prediction of the model (the highest one)
        self.predicted_label = tf.argmax(self.Y_hat, 1)
        # Real class:
        self.real_label = tf.argmax(self.Y_one_hot, 1)
        # Number of correct prediction
        self.correct_prediction = tf.equal(self.predicted_label, self.real_label)
        # Calculate accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.accuracy = tf.mul(100.0, self.accuracy)


    def train(self):
        """
        Train the model
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
                    # Get batch over the dataset
                    location = np.random.choice(range(self.training_size), self.batch_size)
                    batch_xs, batch_ys = self.train_X[location], self.train_Y[location]

                    # Apply the training op
                    (_,
                     loss_train,
                     accuracy_train) = sess.run([self.train_op,
                                                 self.loss,
                                                 self.accuracy],
                                                feed_dict={self.X: batch_xs,
                                                           self.Y: batch_ys})
                    # Model on validation set
                    (loss_val,
                     accuracy_val) = sess.run([self.loss,
                                                 self.accuracy],
                                                feed_dict={self.X: self.val_X,
                                                           self.Y: self.val_Y})

                    # Print loss and accuracy on the batch
                    if s % 200 == 0:
                        print("\033[1;37;40mStep: %04d , "
                          "TRAIN: loss = %.4f - accuracy = %.2f | "
                          "VALIDATION: loss = %.4f - accuracy = %.2f"
                              % ((epoch*self.batch_per_epoch + s),
                             loss_train, accuracy_train,
                             loss_val, accuracy_val) )


            # Do prediction for the test set and write a p.csv file
            Y_hat = sess.run(self.Y_hat,
                             feed_dict={self.X: self.test_X})

            # Write prediction into a file
            csv_pred_file = open('./pred.csv', 'w')
            img_name_file = open('pic_names_test.txt', 'r')
            wr = csv.writer(csv_pred_file)
            wr.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
            for i in range(Y_hat.shape[0]):
                img_name = img_name_file.readline()
                img_name = img_name.rstrip('\n')
                row = [img_name]+list(Y_hat[i,:])
                wr.writerow(row)



def main(_):
    """
    Main function
    :param _:
    :return:
    """

    # Instanciate a Fish classif
    model = Fish_classif(learning_rate=0.01,
                         batch_size=64,
                         num_epoch=10)
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

