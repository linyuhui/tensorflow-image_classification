import os
import tensorflow as tf
import tensorflow.contrib as contrib
import time
import logging
import argparse
import glob
import shutil
import json
from datsaet_factory import get_custom_dataset, CUSTOM_DATASET_INFO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            sleep_secs=1,
                            timeout=None):
    logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
    stop_time = time.time() + timeout if timeout is not None else None
    while True:
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None or checkpoint_path == last_checkpoint:
            if stop_time is not None and time.time() + sleep_secs > \
                    stop_time:
                return None
            time.sleep(sleep_secs)
        else:
            logging.info('Found new checkpoint at %s', checkpoint_path)
            return checkpoint_path


def checkpoints_iterator(checkpoint_dir,
                         min_interval_secs=0,
                         timeout=None,
                         timeout_fn=None):
    checkpoint_path = None
    while True:
        new_checkpoint_path = wait_for_new_checkpoint(
            checkpoint_dir, checkpoint_path, timeout=timeout)
        if new_checkpoint_path is None:
            if not timeout_fn:
                # timed out
                logging.info('Timed-out waiting for a checkpoint.')
                return
            if timeout_fn():
                # The timeout_fn indicated that we are truly done.
                return
            else:
                # The timeout_fn indicated that more checkpoints may come.
                continue
        start = time.time()
        checkpoint_path = new_checkpoint_path
        yield checkpoint_path
        time_to_next_eval = start + min_interval_secs - time.time()
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)


def main(args):
    ckpt_dir = args.ckpt_dir
    model = YourModel()
    dataset = get_custom_dataset(False,
                          data_dir=args.data_dir,
                          batch_size=args.batch_size)
    iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next()
    logits = model(features, False)

    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    preds = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    acc, acc_update_op = tf.metrics.accuracy(labels=labels,
                                             predictions=preds[
                                                 'classes'])
    metrics_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
    metrics_vars_init_op = tf.variables_initializer(
        var_list=metrics_vars, name='metrics_vars_init')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()
    best_acc = 0
    best_acc_history = {}
    for ckpt_path in checkpoints_iterator(ckpt_dir,
                                          min_interval_secs=60,
                                          timeout=300):
        print('Restore from: ', ckpt_path)
        with tf.Session(config=config) as sess:
            sess.run(metrics_vars_init_op)
            saver.restore(sess, ckpt_path)
            sess.run(iterator.initializer)
            step = 0
            total_loss = 0
            while True:
                try:
                    loss_value, _ = sess.run([loss, acc_update_op])
                    total_loss += loss_value
                    step += 1
                except tf.errors.OutOfRangeError:
                    break
            acc_value = sess.run(acc)
            print('Acc {:.4f} Loss{:.4f}\n'.format(
                acc_value, total_loss / step))

            if acc_value > best_acc:
                best_acc = acc_value
                best_step = ckpt_path.split('-')[-1]
                best_acc_history[best_step] = '{:.4f}'.format(best_acc)
                print('Find a better result. Save it')
                for filename in glob.glob(ckpt_path + '.*'):
                    basename = os.path.basename(filename)
                    if not os.path.exists(
                            os.path.join(args.ckpt_dir, 'save')):
                        os.makedirs(
                            os.path.join(args.ckpt_dir, 'save'))
                    shutil.copyfile(
                        filename,
                        os.path.join(args.ckpt_dir, 'save', basename))

                with open(os.path.join(args.ckpt_dir, 'save', 'acc.json'),
                          'w', encoding='utf-8') as fp:
                    json.dump(best_acc_history, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--ckpt_dir', default='...')
    parser.add_argument('--data_dir',
                        default='...')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)
