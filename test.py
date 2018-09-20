# coding:utf-8
# Bin GAO

import os
import cv2
import glob
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',
                    type=str,
                    default='pigdata/test')
parser.add_argument('--model_dir',
                    type=str,
                    default='./model1')
parser.add_argument('--save_dir',
                    type=str,
                    default='./result1')
parser.add_argument('--gpu',
                    type=int,
                    default=0)
flags = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_model():
    file_meta = os.path.join(flags.model_dir, 'model.ckpt.meta')
    file_ckpt = os.path.join(flags.model_dir, 'model.ckpt')

    saver = tf.train.import_meta_graph(file_meta)
    # tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

    sess = tf.InteractiveSession()
    saver.restore(sess, file_ckpt)
    # print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))
    return sess


def read_image(image_path, gray=False):
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main(flags):
    sess = load_model()
    X, mode = tf.get_collection('inputs')
    pred = tf.get_collection('upscore_fuse')[0]

    names=os.listdir(flags.input_dir)
    # names.remove('.DS_Store')
    for name in names:
        inputname=os.path.join(flags.input_dir,name)
        image = read_image(inputname)
        image=cv2.resize(image,(400, 300))
        # sess=tf.InteractiveSession()

        label_pred = sess.run(pred, feed_dict={X: np.expand_dims(image, 0), mode: False})
        merged = np.squeeze(label_pred) * 255
        _, merged = cv2.threshold(merged, 127, 255, cv2.THRESH_BINARY)
        save_name = os.path.join(flags.save_dir, name)
        cv2.imwrite(save_name, merged)
        print('Pred saved')


if __name__ == '__main__':
    main(flags)






