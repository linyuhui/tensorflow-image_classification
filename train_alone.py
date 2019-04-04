import os
import tensorflow as tf
import eco_model
import data_provider
import tensorflow.contrib as contrib
import argparse
import time
import json
from dataset_factory import get_custom_dataset, CUSTOM_DATASET_INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def main(args):
    tf.logging.set_verbosity(tf.logging.ERROR)
    ckpt_dir = args.ckpt_dir
    model = Yourmodel()
    training = True
    dataset = get_custom_dataset(training,
                          data_dir=args.data_dir,
                          batch_size=args.batch_size)
    iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next()
    logits = model(features, training)
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    pred = tf.argmax(input=logits, axis=1, name='classes')
    acc, acc_update_op = tf.metrics.accuracy(labels=labels,
                                             predictions=pred)
    metric_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
    metric_vars_init_op = tf.variables_initializer(metric_vars,
                                                   name='metric_vars_init')

    global_step = tf.train.get_or_create_global_step()
    optim = tf.train.MomentumOptimizer(args.lr, momentum=0.9)
    minimize_op = optim.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        last_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if last_ckpt is not None:
            saver.restore(sess, last_ckpt)
            print('Restore from checkpoint: ', str(last_ckpt.split('-')[-1]))

        global_step_value = None
        for epoch in range(args.num_epochs):
            sess.run(metric_vars_init_op)
            sess.run(iterator.initializer)
            step = 0
            total_loss = 0
            start_time = time.time()
            while True:
                try:
                    _, loss_value, global_step_value, _ = sess.run(
                        [train_op, loss, global_step, acc_update_op]
                    )
                    total_loss += loss_value
                    step += 1
                except tf.errors.OutOfRangeError:
                    break

            acc_value = sess.run(acc)
            msg = 'Epoch {} loss {:.4} train acc {:.4} global step {}'.format(
                epoch, total_loss / step, acc_value, global_step_value)
            msg += '({}s)'.format(time.time() - start_time)
            print(msg)
            saver.save(sess, os.path.join(ckpt_dir, 'model'),
                       global_step=global_step_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=30)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--lr', default=0.00625)
    parser.add_argument('--ckpt_dir', default='./ckpts')
    parser.add_argument('--data_dir',
                        default='...')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args_dict = {}
    for k, v in sorted(vars(args).items()):
        args_dict[k] = str(v)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    with open(os.path.join(args.ckpt_dir, 'cfg.json'),
              'w', encoding='utf-8') as fp:
        json.dump(args_dict, fp, indent=4)

    main(args)


