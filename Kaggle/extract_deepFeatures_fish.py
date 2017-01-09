"""
Extraction of features from Inception v3 model (pretrained on Imagenet)
"""


import os.path
import re
import sys
import tarfile
import time

import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image
import tensorflow as tf

import inception_v3 as inception
from scipy import misc
import re

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('ckpt_path', './inception_v3.ckpt',
                            'The directory where the model was written to or an absolute path to a '
                            'checkpoint file.')
tf.app.flags.DEFINE_string('rgb_dir', '/Users/fabien/Documents/Kaggle/TheNatureConservancyFisheriesMonitoring/Data',
                           """Absolute path to image files.""")
tf.app.flags.DEFINE_string('inception_bottlenecks_dir', '/Users/fabien/Documents/Kaggle/TheNatureConservancyFisheriesMonitoring/Data/inceptionv3_features',
                           """Where to write the extracted bottlenecks.""")



def main(_):
    # Placeholder to feed the image
    images = tf.placeholder(tf.float32, [None, None, None, 3])

    # get the logist and bottlenecks for each images
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(images, num_classes=1001, is_training=False)
        bottlenecks = end_points['PreLogits']
        bottlenecks = tf.squeeze(bottlenecks)

    # List of the classes
    list_classes = os.listdir(os.path.join(FLAGS.rgb_dir, 'train'))
    list_classes = [x for x in list_classes if 'DS' not in x ]

    # Create the directories of the features for each classes
    for c in list_classes:
        if not os.path.exists(os.path.join(FLAGS.inception_bottlenecks_dir, c)):
            os.makedirs(os.path.join(FLAGS.inception_bottlenecks_dir, c))

    print("All the classes: "+str(list_classes))

    # Saver to restore the Inception v3 pretrained model on Imagenet
    saver = tf.train.Saver(max_to_keep=None)

    # # Start the session
    with tf.Session() as sess:
        # Restore the pretrained model
        print("Restoring inception v3 model from: " + FLAGS.ckpt_path)
        saver.restore(sess, FLAGS.ckpt_path)
        print("Restored!")

        # Loop over all the directories (video)
        for label in list_classes:
            list_img = os.listdir(os.path.join(FLAGS.rgb_dir, 'train', label))
            list_img = [x for x in list_img if 'DS' not in x]
            print("Class: " + label + " has " + str(len(list_img)) + " images.")
            for img in list_img:
                print(img)
                # Catch file from directory one by one and convert them to np array
                img_filename = os.path.join(FLAGS.rgb_dir, 'train', label, img)
                image = Image.open(img_filename)
                image = image.resize((299, 299))
                image_data = np.array(image)
                # Inception v3 preprocessing
                image_data = image_data / 255.0
                image_data = image_data - 0.5
                image_data = image_data * 2
                image_data = image_data.reshape((1, 299, 299, 3))

                # Catch the features
                bottlenecks_v = sess.run(bottlenecks, {images: image_data})

                # Save bottlenecks
                txtfile_bottlenecks = os.path.join(FLAGS.inception_bottlenecks_dir, 'train', label, img + '.txt')
                np.savetxt(txtfile_bottlenecks, bottlenecks_v)

        # Finally do the same for test images
        list_test_img = os.listdir(os.path.join(FLAGS.rgb_dir, 'test_stg1'))
        list_test_img = [x for x in list_test_img if 'DS' not in x]
        for img in list_test_img:
            print(img)
            # Catch file from directory one by one and convert them to np array
            img_filename = os.path.join(FLAGS.rgb_dir, 'test_stg1', img)
            image = Image.open(img_filename)
            image = image.resize((299, 299))
            image_data = np.array(image)
            image_data = image_data / 255.0
            image_data = image_data - 0.5
            image_data = image_data * 2
            image_data = image_data.reshape((1, 299, 299, 3))

            # Catch the logits
            bottlenecks_v = sess.run(bottlenecks, {images: image_data})

            # Save bottlenecks
            txtfile_bottlenecks = os.path.join(FLAGS.inception_bottlenecks_dir, 'test_stg1', img + '.txt')
            np.savetxt(txtfile_bottlenecks, bottlenecks_v)


    ################################################
    ## Merge features/labels to one big .txt file ##
    ################################################
    # Training set
    print("Creating training set")
    list_features = []
    list_labels = []
    for c in list_classes:
        print(c)
        list_img_features = os.listdir(os.path.join(FLAGS.inception_bottlenecks_dir,'train', c))
        for img_features in list_img_features:
            features = np.loadtxt(os.path.join(FLAGS.inception_bottlenecks_dir,'train', c, img_features))
            list_features.append(features)
            list_labels.append(list_classes.index(c))
    array_features = np.asarray(list_features)
    array_labels = np.asarray(list_labels)
    # Save to txt files
    np.savetxt('./fish_features.txt', array_features)
    np.savetxt('./fish_labels.txt', array_labels)

    # Test set
    print("Creating test set")
    list_features = []
    list_img_name = []
    list_img_features = os.listdir(os.path.join(FLAGS.inception_bottlenecks_dir,'test_stg1'))
    for img_features in list_img_features:
        features = np.loadtxt(os.path.join(FLAGS.inception_bottlenecks_dir,'test_stg1', img_features))
        list_features.append(features)
        list_labels.append(list_classes.index(c))
        list_img_name.append(img_features.split('.')[0] + '.jpg')
    array_features = np.asarray(list_features)
    np.savetxt('./fish_features_test.txt', array_features)

    # Write name of each pictures for submission
    with open('./pic_names_test.txt', 'w') as thefile:
        for item in list_img_name:
            thefile.write("%s\n" % item)


if __name__ == '__main__':
  tf.app.run()
