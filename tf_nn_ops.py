import tensorflow as tf
if tf.__version__ == '1.8.0':
    from tensorflow.python.layers.base import _unique_layer_name as unique_layer_name
else: # such as 1.12, 1.13
    from tensorflow.python.keras.engine.base_layer_utils import unique_layer_name
from tensorflow.python.ops.init_ops import _compute_fans
import math


def avg_pool2d(inputs, pool_size, stride=1, padding=0,
               data_format='channels_last', name=None):
    if data_format == 'channels_last':
        padded_inputs = tf.pad(inputs,
                               paddings=[[0, 0], [padding, padding],
                                         [padding, padding], [0, 0]])
    else:
        padded_inputs = tf.pad(inputs,
                               paddings=[[0, 0], [0, 0],
                                         [padding, padding],
                                         [padding, padding]])

    return tf.layers.average_pooling2d(
        padded_inputs, pool_size, stride, name=name)


def conv2d(inputs, filters, kernel_size, stride=1, padding=0,
           groups=1, rate=1,
           use_bias=True, initializer=tf.variance_scaling_initializer(),
           data_format='channels_last', name='conv2d'):
    if data_format == 'channels_last':
        data_format = 'NHWC'
        in_channels = inputs.get_shape().as_list()[-1]
    elif data_format == 'channels_first':
        data_format = 'NCHW'
        in_channels = inputs.get_shape().as_list()[1]
    else:
        raise ValueError('Invalid data format.')

    name = unique_layer_name(name)
    with tf.variable_scope(name):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            kernel_size = [kernel_size[0], kernel_size[1], in_channels / groups,
                           filters]
        else:
            kernel_size = [kernel_size, kernel_size, in_channels / groups,
                           filters]

        if isinstance(stride, list) or isinstance(stride, tuple):
            stride = [1, stride[0], stride[1],
                      1] if data_format == 'NHWC' else [1, 1, stride[0],
                                                        stride[1]]
        else:
            stride = [1, stride, stride,
                      1] if data_format == 'NHWC' else [1, 1, stride,
                                                        stride]

        if isinstance(padding, int):
            padding = [padding, padding]

        if data_format == 'NHWC':
            padded_inputs = tf.pad(inputs, tf.constant(
                [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]],
                 [0, 0]]))
        else:
            padded_inputs = tf.pad(inputs, tf.constant(
                [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]],
                 [0, 0]]))

        weight = tf.get_variable('weight', kernel_size, initializer=initializer)

        if use_bias:
            fan_in, _ = _compute_fans(weight.get_shape().as_list())
            bound = 1 / math.sqrt(fan_in)
            bias = tf.get_variable('bias', [filters],
                                   initializer=tf.initializers.random_uniform(
                                       -bound, bound))
        else:
            bias = None

        channel_axis = 3 if data_format == 'NHWC' else 1
        if isinstance(rate, int):
            rate = [1, rate, rate, 1]
        if groups == 1:
            outputs = tf.nn.conv2d(padded_inputs, weight, stride, 'VALID',
                                   dilations=rate, data_format=data_format)
        else:
            inputs = tf.split(padded_inputs, groups, channel_axis)
            kernels = tf.split(weight, groups, 3)
            outputs = [
                tf.nn.conv2d(i, k, stride, 'VALID', dilations=rate,
                             data_format=data_format) for
                i, k in zip(inputs, kernels)]
            outputs = tf.concat(outputs, channel_axis)

        outputs = tf.nn.bias_add(
            outputs, bias,
            data_format=data_format) if use_bias else outputs
        return outputs


def batch_norm(inputs, training, data_format='channels_last'):
    return tf.layers.batch_normalization(
        inputs, axis=3 if data_format == 'channels_last' else 1,
        momentum=0.997, epsilon=1e-5, center=True,
        scale=True, training=training, fused=True
    )


def conv2d_with_bn(inputs, training, filters, stride,
                   data_format='channels_last',
                   name='conv2d_with_bn'):
    name = unique_layer_name(name)
    with tf.variable_scope(name):
        net = conv2d(inputs, filters, kernel_size=3, stride=stride,
                     padding=1, use_bias=False)
        net = batch_norm(net, training)
        net = tf.nn.relu(net)
        return net


def separable_conv2d(inputs, training, filters, stride,
                     data_format='channels_last',
                     name='separable_conv2d'):
    assert data_format == 'channels_last'
    in_channels = inputs.get_shape().as_list()[-1]

    name = unique_layer_name(name)
    with tf.variable_scope(name):
        # depthwise
        net = conv2d(inputs, in_channels, kernel_size=3, stride=stride,
                     padding=1,
                     groups=in_channels, use_bias=False)
        net = batch_norm(net, training)
        net = tf.nn.relu(net)

        # pointwise
        net = conv2d(net, filters, 1, 1, use_bias=False)
        net = batch_norm(net, training)
        net = tf.nn.relu(net)

        return net


if __name__ == '__main__':
    name = unique_layer_name('conv')
    print(name)
