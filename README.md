# Salience Object Detection
Salience Object Detection Using python + tensorflow This is tensorflow implementation for cvpr2017 paper "Deeply Supervised Salient Object Detection with Short Connections".<br>

Cuz i trained this model on my company's data,so to be honest i dont know this model still reach the performance mentioned in paper or not.

## Project info
This code can be used to train, but the data is owned by my company.I'll try my best to provide code and model that can do inference.<br>
<br>
My Chinese blog about the implementation of this paper http://blog.csdn.net/gbyy42299/article/details/79427457  <br>

## Training on your own dataset
I trained this model on gpu Tesla P100(1200 images, 10h)<br> 
1.Just put ur data under the folder named 'dataset'.<br> 
2.run csv_generater.py to get the csv files for ur training.<br> 
3.run train.py for training ur own model.<br> 
4.run test.py for testing new photo.<br>
<br>
## Tensorboard info
<br>
![](https://github.com/gbyy422990/salience_object_detection/blob/master/tensorboard/%E6%9C%AA%E5%91%BD%E5%90%8D.png)
<br>
![](https://github.com/gbyy422990/salience_object_detection/blob/master/tensorboard/%E6%9C%AA%E5%91%BD%E5%90%8D%203.png)

