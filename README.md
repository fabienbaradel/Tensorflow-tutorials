# Introduction to Deep Learning with Tensorflow

Tutorials with Tensorflow implementations. Initially designed for [ENSAI SID 2017](http://www.ensai.fr/formation/id-3e-annee-ingenieur/filiere-statistique-et-ingenierie-des-donnees.html).

A .ova file (Ubuntu 16.04 - ~ 3.5 Go) is available on a USB key where Python 2.7 and Tensorflow (0.12.0) are already installed.
Go to VirtualBox and 'import a virtual environment' and select the .ova file.

Slides available on [my website](https://fabienbaradel.github.io): ["Intro to deep learning with Tensorflow"](https://fabienbaradel.github.io/images/tensorflow_ensai_SID_13_01_17.pdf)

Feel free to install directly Tensorflow on your laptop [TensorFlow Installation Guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)


## Introduction
- Hello World ([code](Intro/hello_world.py))
- Basic Maths ([code](Intro/math_ops.py))

## Understanding Stochastic Gradient Descent
- Linear Regression ([exo](SGD/linear_regression_exo.py)) - ([solution](SGD/linear_regression.py))
- Binary Classification ([exo](SGD/binary_classifcation_exo.py)) - ([solution](SGD/binary_classifcation.py))

## MNIST
- Softmax ([exo](MNIST/softmax_exo.py)) - ([solution](MNIST/softmax.py))
- Mulilayer Perceptron ([exo](MNIST/mlp_exo.py)) - ([solution](MNIST/mlp.py))
- One Conv + Max Pool ([exo](MNIST/one_conv_exo.py)) - ([solution](MNIST/one_conv.py))
- Make your life easier => [SLIM](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) ;)
- LeNet ([exo](MNIST/lenet_exo.py)) - ([solution](MNIST/lenet.py))
- Autoencoder ([exo](MNIST/autoencoder_exo.py)) - ([solution](MNIST/autoencoder.py))
- Conv-Deconv Autoencoder ([exo](MNIST/conv_ae_exo.py)) - ([solution](MNIST/conv_ae.py))
- RNN ([exo](MNIST/rnn_exo.py)) - ([solution](MNIST/rnn_exo.py))
- GAN (coming...)
- DCGAN (coming...)


## The Nature Conservancy Fisheries Monitoring ([Kaggle](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring))
- Have a look at the [README](Kaggle/README.md) of this repository first
- Inception v3 Features Extraction ([code](Kaggle/extract_deepFeatures_fish.py))
- Classification from DeepFeatures ([exo](Kaggle/classif_fish_exo.py)) - ([softmax regression](kaggle/classif_fish.py))


## Frozen Lake
- Q-Learning with neural network ([exo](RL/q_learning_neural_net_exo.py)) - ([solution](RL/q_learning_neural_net.py))

## Thanks
Code inspired by other great tutorials from [@aymericdamien](https://github.com/aymericdamien/TensorFlow-Examples) (SGD and MNIST parts), [@awjuliani](https://github.com/awjuliani/DeepRL-Agents) (RL part), [@pkmital](https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py) (conv-deconv autoencoder).  
Here are my favorite deep learning blogs: [colah.github.io](http://colah.github.io/), [karpathy.github.io](http://karpathy.github.io/), [wildml.com](http://www.wildml.com/)



## Dependencies
```
tensorflow (>0.12.1)
numpy
matplotlib
gym (Frozen Lake only)
```

An update one the repo will be done soone (python 3.5 and Tensorflow 1.2)
