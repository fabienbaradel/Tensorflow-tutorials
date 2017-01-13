# Introduction to Deep Learning with Tensorflow

Tutorials with Tensorflow implementations. Specially designed for [ENSAI SID 2017](http://www.ensai.fr/formation/id-3e-annee-ingenieur/filiere-statistique-et-ingenierie-des-donnees.html).

A .ova file (Ubuntu 16.04 - ~ 3.5 Go) is available on a USB key where Python 2.7 and Tensorflow (0.12.0) are already installed.
Go to VirtualBox and 'import a virtual environment' and select the .ova file.

Slides: ["Intro to deep learning with Tensorflow"](https://fabienbaradel.github.io/images/tensorflow_ensai_SID_13_01_17.pdf)

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
- RNN ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/rnn_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/MNIST/rnn_exo.py))

## The Nature Conservancy Fisheries Monitoring ([Kaggle](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring))
- Have a look at the [README](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/Kaggle/README.md) of this repository first
- Inception v3 Features Extraction ([code](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/Kaggle/extract_deepFeatures_fish.py))
- Classification from DeepFeatures ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/Kaggle/classif_fish_exo.py)) - ([softmax regression](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/kaggle/classif_fish.py))


## Frozen Lake
- Q-Learning with neural network ([exo](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/RL/q_learning_neural_net_exo.py)) - ([solution](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/RL/q_learning_neural_net.py))



## Thanks
Code inspired by other great tutorials from [@aymericdamien](https://github.com/aymericdamien/TensorFlow-Examples), [@awjuliani](https://github.com/awjuliani/DeepRL-Agents), [@pkmital](https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py).  
Do not hesitate to read more about deep learning on those awesome blogs: [colah.github.io](http://colah.github.io/), [karpathy.github.io](http://karpathy.github.io/), [wildml.com](http://www.wildml.com/)



## Dependencies
```
tensorflow
numpy
matplotlib
```
