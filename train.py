# Yhlin
import os
import tensorflow as tf
import argparse
import time
import json
import glob
import shutil
from dataset_factory import get_custom_dataset, CUSTOM_DATASET_INFO
from resnet_model import ResNet, get_block_sizes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_optimizer(optim_name, lr):
    if optim_name == 'adam':
        return tf.train.AdamOptimizer(lr)
    elif optim_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr)
    elif optim_name == 'momentum':
        return tf.train.MomentumOptimizer(lr, momentum=0.9)
    else:
        return tf.train.GradientDescentOptimizer(lr)


def get_learning_rate(global_step, batch_size, num_images,
                      boundary_epochs, decay_rates,
                      base_lr=0.1, warmup=False, decay=False):
    batches_per_epoch = num_images / batch_size
    if decay:
        boundaries = [int(batches_per_epoch * epoch) for epoch in
                      boundary_epochs]
        vals = [base_lr * rate for rate in decay_rates]
        lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    else:
        lr = tf.convert_to_tensor(base_lr, tf.float32)
    if warmup:
        warmup_steps = int(batches_per_epoch * 5)
        warmup_lr = (base_lr * tf.cast(global_step, tf.float32) /
                     tf.cast(warmup_steps, tf.float32))
        return tf.cond(pred=global_step < warmup_steps,
                       true_fn=lambda: warmup_lr,
                       false_fn=lambda: lr)
    return lr


def reset_last_ckpt_dir(ckpt_dir):
    count = 1
    ckpt_dir += '_0'
    while os.path.exists(ckpt_dir):
        index = ckpt_dir.rfind('_')
        ckpt_dir = ckpt_dir[:index] + '_{}'.format(count)
        count += 1
    count -= 1
    index = ckpt_dir.rfind('_')
    ckpt_dir = ckpt_dir[:index] + '_{}'.format(count)
    shutil.rmtree(ckpt_dir)
    return ckpt_dir


def prepare_ckpt_dir(ckpt_dir):
    count = 1
    ckpt_dir += '_0'
    while os.path.exists(ckpt_dir):
        index = ckpt_dir.rfind('_')
        ckpt_dir = ckpt_dir[:index] + '_{}'.format(count)
        count += 1
    return ckpt_dir


def main(args):
    ckpt_dir = args.ckpt_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    data_dir = args.data_dir
    boundaries = [int(x) for x in args.boundaries.split(',')]
    decay_rates = [float(x) for x in args.decay_rates.split(',')]

    training = tf.placeholder(tf.bool, shape=[], name='training')

    # Dataset
    train_dataset = get_custom_dataset(True,
                                                    data_dir=data_dir,
                                                    batch_size=batch_size)
    val_dataset = get_custom_dataset(False, data_dir=data_dir,
                                                  batch_size=batch_size)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    features, labels = iterator.get_next()
    train_ds_init_op = iterator.make_initializer(train_dataset)
    val_ds_init_op = iterator.make_initializer(val_dataset)

    # Model
    resnet_size = 50
    model = ResNet(resnet_size=resnet_size,
                   num_classes=CUSTOM_DATASET_INFO.num_classes,
                   num_filters=64,
                   kernel_size=7,
                   conv_stride=2,
                   first_pool_size=3,
                   first_pool_stride=2,
                   block_sizes=get_block_sizes(resnet_size),
                   block_strides=[1, 2, 2, 2],
                   resnet_version=2,
                   data_format='channels_last',
                   dtype=tf.float32)
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
    lr = get_learning_rate(global_step, batch_size,
                           CUSTOM_DATASET_INFO.train,
                           boundary_epochs=boundaries,
                           decay_rates=decay_rates,
                           base_lr=args.lr,
                           warmup=args.warmup,
                           decay=args.decay)

    tf.identity(lr, name='learning_rate')

    optimizer = get_optimizer(args.optim_name, lr)
    minimize_op = optimizer.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()
    best_acc_history = {}
    best_acc = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        last_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if last_ckpt is not None:
            saver.restore(sess, last_ckpt)
            print('Restore from checkpoint: ', str(last_ckpt.split('-')[-1]))

        global_step_value = None
        for epoch in range(num_epochs):
            # Train
            sess.run(metric_vars_init_op)
            sess.run(train_ds_init_op)
            step = 0
            total_loss = 0
            start_time = time.time()
            lr_value = 0
            while True:
                try:
                    _, _, loss_value, global_step_value, lr_value = sess.run(
                        [train_op, acc_update_op, loss, global_step, lr],
                        feed_dict={training: True}
                    )
                    total_loss += loss_value
                    step += 1
                except tf.errors.OutOfRangeError:
                    break

            acc_value = sess.run(acc)
            msg = 'Epoch: {}\n'.format(epoch)
            msg += '  Train: Loss {:.4f} Acc {:.4f} global step {}'.format(
                total_loss / step, acc_value, global_step_value)
            msg += ' lr {:.6f}'.format(lr_value)
            msg += '({:.2f}s)'.format(time.time() - start_time)
            print(msg)
            saver.save(sess, os.path.join(ckpt_dir, 'model'),
                       global_step=global_step_value)

            # Eval
            sess.run(metric_vars_init_op)
            sess.run(val_ds_init_op)
            step = 0
            total_loss = 0
            while True:
                try:
                    loss_value, _ = sess.run(
                        [loss, acc_update_op],
                        feed_dict={training: False})
                    total_loss += loss_value
                    step += 1
                except tf.errors.OutOfRangeError:
                    break
            acc_value = sess.run(acc)
            print('  Valid: Loss {:.4f} Acc {:.4f} '.format(
                total_loss / step, acc_value))
            last_ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if acc_value > best_acc:
                best_acc = acc_value
                best_step = last_ckpt.split('-')[-1]
                best_acc_history[best_step] = '{:.4f}'.format(best_acc)
                print('    Gets a better result. Save it.')
                for filename in glob.glob(last_ckpt + '.*'):
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
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--lr', type=float, default=0.00625)
    parser.add_argument('--ckpt_dir', default='...')
    parser.add_argument('--data_dir',
                        default='...')
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--decay', action='store_true', default=False)
    parser.add_argument('--boundaries', default='10,20,30,40')
    parser.add_argument('--decay_rates', default='1,0.1,0.01,0.001,0.0001')
    parser.add_argument('--optim_name', type=str, default='momentum')
    parser.add_argument('--go_on', action='store_true', default=False)
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.reset:
        assert not args.go_on
        args.ckpt_dir = reset_last_ckpt_dir(args.ckpt_dir)



    if not args.go_on:
        args.ckpt_dir = prepare_ckpt_dir(args.ckpt_dir)

    args_dict = {}
    for k, v in sorted(vars(args).items()):
        args_dict[k] = str(v)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    with open(os.path.join(args.ckpt_dir, 'cfg.json'),
              'w', encoding='utf-8') as fp:
        json.dump(args_dict, fp, indent=4)

    main(args)
