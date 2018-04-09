#coding:utf-8
#Bin GAO

import os
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import model
import time

from model import train_op
from model import loss_CE,loss_IOU



h = 1200   #4032
w = 1600   #3024
c_image = 3
c_label = 1
g_mean = [142.53,129.53,120.20]

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default = './pig1.csv')

parser.add_argument('--test_dir',
                    default = './pigtest1.csv')

parser.add_argument('--model_dir',
                    default = './model1')

parser.add_argument('--epochs',
                    type = int,
                    default = 10)

parser.add_argument('--peochs_per_eval',
                    type = int,
                    default = 1)

parser.add_argument('--logdir',
                    default = './logs1')

parser.add_argument('--batch_size',
                    type = int,
                    default = 1)

parser.add_argument('--is_cross_entropy',
                    action = 'store_true',
                    default=True)

parser.add_argument('--learning_rate',
                    type = float,
                    default = 1e-3)

#衰减系数
parser.add_argument('--decay_rate',
                    type = float,
                    default = 0.9)

#衰减速度model
parser.add_argument('--decay_step',
                    type = int,
                    default = 100)

parser.add_argument('--weight',
                    nargs = '+',
                    type = float,
                    default = [1.0,1.0])

parser.add_argument('--random_seed',
                    type = int,
                    default = 1234)

parser.add_argument('--gpu',
                    type = str,
                    default = 1)

flags = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_config():

    ''''#允许增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    '''

    #控制使用率
    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
    config = tf.ConfigProto(gpu_options = gpu_options)
    session = tf.Session(config=config)


def data_augmentation(image,label,training=True):
    if training:
        image_label = tf.concat([image,label],axis = -1)
        print('image label shape concat',image_label.get_shape())

        maybe_flipped = tf.image.random_flip_left_right(image_label)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)
        #maybe_flipped = tf.random_crop(maybe_flipped,size=[h/2,w/2,image_label.get_shape()[-1]])


        image = maybe_flipped[:, :, :-1]
        mask = maybe_flipped[:, :, -1:]

        #image = tf.image.random_brightness(image, 0.7)
        #image = tf.image.random_hue(image, 0.3)
        #设置随机的对比度
        #tf.image.random_contrast(image,lower=0.3,upper=1.0)

        return image, mask

def read_csv(queue,augmentation=True):
    #csv = tf.train.string_input_producer(['./data/train/csv','./data/test.csv'])
    csv_reader = tf.TextLineReader(skip_header_lines=1)

    _, csv_content = csv_reader.read(queue)

    image_path, label_path = tf.decode_csv(csv_content,record_defaults=[[""],[""]])

    image_file = tf.read_file(image_path)
    label_file = tf.read_file(label_path)

    image = tf.image.decode_jpeg(image_file, channels = 3)
    image.set_shape([h,w,c_image])
    image = tf.cast(image, tf.float32)

    label = tf.image.decode_jpeg(label_file, channels = 1)
    label.set_shape([h,w,c_label])

    label = tf.cast(label,tf.float32)
    label = label / (tf.reduce_max(label) + 1e-7)

    #数据增强
    if augmentation:
        image,label = data_augmentation(image,label)
    else:
        pass
    return image,label

def main(flags):
    current_time = time.strftime("%m/%d/%H/%M/%S")
    train_logdir = os.path.join(flags.logdir, "pig", current_time)
    test_logdir = os.path.join(flags.logdir, "test", current_time)

    train = pd.read_csv(flags.data_dir)

    num_train = train.shape[0]

    test = pd.read_csv(flags.test_dir)
    num_test = test.shape[0]

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = [flags.batch_size,h,w,c_image],name = 'X')
    y = tf.placeholder(tf.float32,shape = [flags.batch_size,h,w,c_label], name = 'y')
    mode = tf.placeholder(tf.bool, name='mode')

    score_dsn6_up, score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse = model.unet(X,mode)

    #print(score_dsn6_up.get_shape().as_list())

    loss6 = loss_CE(score_dsn6_up, y)
    loss5 = loss_CE(score_dsn5_up, y)
    loss4 = loss_CE(score_dsn4_up, y)
    loss3 = loss_CE(score_dsn3_up, y)
    loss2 = loss_CE(score_dsn2_up, y)
    loss1 = loss_CE(score_dsn1_up, y)
    loss_fuse = loss_CE(upscore_fuse, y)
    tf.summary.scalar("CE6", loss6)
    tf.summary.scalar("CE5", loss5)
    tf.summary.scalar("CE4", loss4)
    tf.summary.scalar("CE3", loss3)
    tf.summary.scalar("CE2", loss2)
    tf.summary.scalar("CE1", loss1)
    tf.summary.scalar("CE_fuse", loss_fuse)


    Loss=loss6+loss5+loss4+loss3+loss2+2*loss1+loss_fuse
    tf.summary.scalar("CE_total", Loss)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step,
                                               decay_steps=flags.decay_step,
                                               decay_rate=flags.decay_rate, staircase=True)

    with tf.control_dependencies(update_ops):
        training_op = train_op(Loss,learning_rate)


    train_csv = tf.train.string_input_producer(['pig1.csv'])
    test_csv = tf.train.string_input_producer(['pigtest1.csv'])

    train_image, train_label = read_csv(train_csv,augmentation=True)
    test_image, test_label = read_csv(test_csv,augmentation=False)

    #batch_size是返回的一个batch样本集的样本个数。capacity是队列中的容量
    X_train_batch_op, y_train_batch_op = tf.train.shuffle_batch([train_image, train_label],batch_size = flags.batch_size,
                                              capacity = flags.batch_size*5,min_after_dequeue = flags.batch_size*2,
                                              allow_smaller_final_batch = True)

    X_test_batch_op, y_test_batch_op = tf.train.batch([test_image, test_label],batch_size = flags.batch_size,
                                                        capacity = flags.batch_size*2,allow_smaller_final_batch = True)



    print('Shuffle batch done')
    #tf.summary.scalar('loss/Cross_entropy', CE_op)
    score_dsn6_up = tf.nn.sigmoid(score_dsn6_up)
    score_dsn5_up = tf.nn.sigmoid(score_dsn5_up)
    score_dsn4_up = tf.nn.sigmoid(score_dsn4_up)
    score_dsn3_up = tf.nn.sigmoid(score_dsn3_up)
    score_dsn2_up = tf.nn.sigmoid(score_dsn2_up)
    score_dsn1_up = tf.nn.sigmoid(score_dsn1_up)
    upscore_fuse = tf.nn.sigmoid(upscore_fuse)
    print(upscore_fuse.get_shape().as_list())

    tf.add_to_collection('inputs', X)
    tf.add_to_collection('inputs', mode)
    tf.add_to_collection('score_dsn6_up', score_dsn6_up)
    tf.add_to_collection('score_dsn5_up', score_dsn5_up)
    tf.add_to_collection('score_dsn4_up', score_dsn4_up)
    tf.add_to_collection('score_dsn3_up', score_dsn3_up)
    tf.add_to_collection('score_dsn2_up', score_dsn2_up)
    tf.add_to_collection('score_dsn1_up', score_dsn1_up)
    tf.add_to_collection('upscore_fuse', upscore_fuse)


    tf.summary.image('Input Image:', X)
    tf.summary.image('Label:', y)
    tf.summary.image('score_dsn6_up:', score_dsn6_up)
    tf.summary.image('score_dsn5_up:', score_dsn5_up)
    tf.summary.image('score_dsn4_up:', score_dsn4_up)
    tf.summary.image('score_dsn3_up:', score_dsn3_up)
    tf.summary.image('score_dsn2_up:', score_dsn2_up)
    tf.summary.image('score_dsn1_up:', score_dsn1_up)
    tf.summary.image('upscore_fuse:', upscore_fuse)

    tf.summary.scalar("learning_rate", learning_rate)

    # 添加任意shape的Tensor，统计这个Tensor的取值分布
    tf.summary.histogram('score_dsn1_up:', score_dsn1_up)
    tf.summary.histogram('score_dsn2_up:', score_dsn2_up)
    tf.summary.histogram('score_dsn3_up:', score_dsn3_up)
    tf.summary.histogram('score_dsn4_up:', score_dsn4_up)
    tf.summary.histogram('score_dsn5_up:', score_dsn5_up)
    tf.summary.histogram('score_dsn6_up:', score_dsn6_up)
    tf.summary.histogram('upscore_fuse:', upscore_fuse)


    #添加一个操作，代表执行所有summary操作，这样可以避免人工执行每一个summary op
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_writer = tf.summary.FileWriter(test_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        # if not os.listdir(flags.model_dir):
        #     print('No model')
        #     try:
        #         os.rmdir(flags.model_dir)
        #     except Exception as e:
        #         print(e)
        #     os.mkdir(flags.model_dir)
        # else:
        #     latest_check_point = tf.train.latest_checkpoint(flags.model_dir)
        #     saver.restore(sess, latest_check_point)
        if os.path.exists(flags.model_dir) and tf.train.checkpoint_exists(flags.model_dir):
            latest_check_point = tf.train.latest_checkpoint(flags.model_dir)
            saver.restore(sess, latest_check_point)

        else:
            print('No model')
            try:
                os.rmdir(flags.model_dir)
            except Exception as e:
                print(e)
            os.mkdir(flags.model_dir)

        try:
            #global_step = tf.train.get_global_step(sess.graph)

            #使用tf.train.string_input_producer(epoch_size, shuffle=False),会默认将QueueRunner添加到全局图中，
            #我们必须使用tf.train.start_queue_runners(sess=sess)，去启动该线程。要在session当中将该线程开启,不然就会挂起。然后使用coord= tf.train.Coordinator()去做一些线程的同步工作,
            #否则会出现运行到sess.run一直卡住不动的情况。
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(flags.epochs):
                for step in range(0,num_train,flags.batch_size):
                    X_train, y_train = sess.run([X_train_batch_op,y_train_batch_op])
                    _, step_ce, step_summary, global_step_value = sess.run([training_op, Loss, summary_op, global_step],
                                                                           feed_dict={X: X_train, y: y_train,
                                                                                      mode: True})

                    train_writer.add_summary(step_summary, global_step_value)
                    print('epoch:{} step:{} loss_CE:{}'.format(epoch + 1, global_step_value, step_ce))
                for step in range(0,num_test,flags.batch_size):
                    X_test, y_test = sess.run([X_test_batch_op, y_test_batch_op])
                    step_ce, step_summary = sess.run([Loss, summary_op], feed_dict={X: X_test, y: y_test, mode: False})

                    test_writer.add_summary(step_summary, epoch * (
                    num_train // flags.batch_size) + step // flags.batch_size * num_train // num_test)
                    print('Test loss_CE:{}'.format(step_ce))
                saver.save(sess, '{}/model.ckpt'.format(flags.model_dir))

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.model_dir))


if __name__ == '__main__':
    #set_config()
    main(flags)
