Few files for a first submission to [The Natue Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)

We use a deep learning pretrained models (already trained on Imagenet) and extract deep Features from it.
Each images is now summarize by a vector of size 2048. This part is done by [here](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/Kaggle/classif_fish.py).
You have to download the image from the Kaggle website to do it (Don't do it by yourself it takes a while...)
The pretrained inception v3 checkpoint can be download [here](https://drive.google.com/file/d/0B3K4bVd6ydRwVzRqLUJvVTJTSTA/view?usp=sharing).

In your virtual machine you have 4 .txt in this repository with the labels/features/image_name of the Fish dataset:
- fish_features.txt: inception_v3 features of the training set
- fish_labels.txt: labels of the training set
- fish_features_test.txt: inception_v3 features of the test set
- pic_names_test.py: name of the pictures of the test set
The .txt files can be download [here](https://drive.google.com/file/d/0B3K4bVd6ydRwR1VrVkwtblJnNHc/view?usp=sharing) if you are not working with the VM. Don't forget to add them in this directory on your laptop.

Complete the [classif_fish_exo.py](https://github.com/fabienbaradel/Tensorflow-tutorials/blob/master/Kaggle/classif_fish.py).
You will create a classification from the dee features. Feel free to build the network you want.
Running this code will create a pred.csv file in this directory which corresponds to the submission file required by Kaggle.

A simple softmax regression with 10 epochs leads to a score of ~1.14 (rank: 300/1100).
Improve it and submit the pred.csv in Kaggle! ;)