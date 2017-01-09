# Introduction to Deep Learning with Tensorflow

Tutorials with Tensorflow implementations. Specially designed for [ENSAI SID 2017](http://www.ensai.fr/formation/id-3e-annee-ingenieur/filiere-statistique-et-ingenierie-des-donnees.html).

A [.ova](https://drive.google.com/file/d/0B3K4bVd6ydRwdFlUU3NEYm93bm8/view?usp=sharing) file (Ubuntu 16.04 - ~ 4 Go) is available on a USB key where Anaconda and Tensorflow (0.12.0) are already installed.
Go to VirtualBox and 'import a virtual environment' and select the .ova file.

Slides: [Intro to deep learning with Tensorflow](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/introduction_to_deep_learning_with_tensorflow.pdf)

Feel free to install directly Tensorflow on your laptop [TensorFlow Installation Guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)


## Introduction
- Hello World ([code](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/Intro/hello_world.py))
- Basic Maths ([code](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/Intro/math_ops.py))

## Understanding Stochastic Gradient Descent
- Linear Regression ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/SGD/linear_regression_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/SGD/linear_regression.py))
- Binary Classification ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/SGD/binary_classifcation_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/SGD/binary_classifcation.py))

## MNIST
- Softmax ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/softmax_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/softmax.py))
- Mulilayer Perceptron ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/mlp_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/mlp.py))
- One Conv + Max Pool ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/one_conv_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/one_conv.py))
- Make your life easier => [SLIM](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) ;)
- LeNet ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/lenet_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/lenet.py))
- Autoencoder ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/autoencoder_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/autoencoder.py))
- Conv-Deconv Autoencoder ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/conv_ae_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/conv_ae.py))
- GAN  (coming)
- DCGAN (coming)
- RNN ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/rnn_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/rnn_exo.py))

## The Nature Conservancy Fisheries Monitoring ([Kaggle](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring))
- Inception v3 Features Extraction ([code](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/kaggle/extract_deepFeatures_fish.py))
- Classification from DeepFeatures ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/kaggle/classif_fish_exo.py)) - ([softmax regression](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/kaggle/classif_fish.py))


## Dependencies
```
tensorflow
numpy
matplotlib
```
