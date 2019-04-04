import os
import tensorflow as tf
import time
import logging
import argparse
import glob
import shutil
import json
from dataset_factory import get_cars_dataset_from_record, CAR_RECORD_INFO
import mobilenet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_CLASSES = 2
DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
CLASS_NAME = ['0', '1']
NUM_IMAGES = {
    'train': 6255,
    'val': 330,
}
CHANNEL_MEANS = [0.485, 0.456, 0.406]
CHANNEL_STDS = [0.229, 0.224, 0.225]
DATASET_NAME = 'vapd'
EVAL_DIRNAME = 'test'
NUM_FRAMES = 6
INPUT_SIZE = (224, 224)


def main(args):
    model = mobilenet.MobileNet(num_classes=CAR_RECORD_INFO.num_classes)
    dataset = get_cars_dataset_from_record(False,
                                           data_dir=args.data_dir,
                                           batch_size=args.batch_size,
                                           val_prefix='validation')
    iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next()
    logits = model(features, False)

    preds = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    acc, acc_update_op = tf.metrics.accuracy(labels=labels,
                                             predictions=preds[
                                                 'classes'])
    metric_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
    metric_vars_init_op = tf.variables_initializer(
        var_list=metric_vars, name='metrics_vars_init')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, args.ckpt)
        sess.run(iterator.initializer)
        sess.run(metric_vars_init_op)
        while True:
            try:
                _, logits_value = sess.run(
                    [acc_update_op, logits])
            except tf.errors.OutOfRangeError:
                break

        acc_value = sess.run(acc)
        print('Acc {:.4f} '.format(acc_value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--ckpt',
                        default='.../car_ckpts/model-70146')
    parser.add_argument('--data_dir',
                        default='.../tfrecord')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)
