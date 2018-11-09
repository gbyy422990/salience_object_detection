# Salience Object Detection
Salience Object Detection Using python + tensorflow This is tensorflow implementation for cvpr2017 paper "Deeply Supervised Salient Object Detection with Short Connections".<br>

If it is useful for u, pls give me a star! THX <br>

Cuz i trained this model on my company's data,so to be honest i dont know this model still reach the performance mentioned in paper or not.

## Project info
This code can be used to train, but the data is owned by my company.I'll try my best to provide code and model that can do inference.<br>
<br>
My Chinese blog about the implementation of this paper http://blog.csdn.net/gbyy42299/article/details/79427457  <br>

## Usage
I trained this model on gpu Tesla P100(1200 images, 10h)<br> 
u need download vgg16.npy as pre-trained model.<br>
1.Just put ur data under the folder named 'dataset'.<br> 
2.run csv_generater.py to get the csv files for ur training.<br> 
3.run train.py for training ur own model.<br> 
4.run test.py for testing new photo.<br>
<br>
## Tensorboard info
<img width="480" height="360" src="https://github.com/gbyy422990/salience_object_detection/blob/master/tensorboard/%E6%9C%AA%E5%91%BD%E5%90%8D.png"/>
<br>
<img width="480" height="360" src="https://github.com/gbyy422990/salience_object_detection/blob/master/tensorboard/%E6%9C%AA%E5%91%BD%E5%90%8D%203.png"/>
<br>

# Update
## 2018.02.04
I added model scripy fot u guys for training on ur own dataset.
## 2018.03.28
I fixed some bug, like loss problem etc, now u can use it.
## 2018.04.13
U can find another salience obeject detection in this url: https://github.com/gbyy422990/salience_object-detection-non-local
