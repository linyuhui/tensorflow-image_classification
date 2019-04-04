import tensorflow as tf


def conv2d(inputs, filters, kernel_size, strides, padding,
           data_format='channels_last'):
    if data_format == 'channels_last':
        padded_inputs = tf.pad(inputs,
                               paddings=[[0, 0], [padding, padding],
                                         [padding, padding], [0, 0]])
    else:
        padded_inputs = tf.pad(inputs,
                               paddings=[[0, 0], [0, 0],
                                         [padding, padding],
                                         [padding, padding]])
    return tf.layers.conv2d(
        padded_inputs, filters, kernel_size, strides,
        use_bias=True, kernel_initializer=tf.variance_scaling_initializer(),
        bias_initializer=tf.zeros_initializer(),
        data_format=data_format
    )


def batch_norm(inputs, training, data_format):
    return tf.layers.batch_normalization(
        inputs, axis=3 if data_format == 'channels_last' else 1,
        momentum=0.997, eposilon=1e-5, center=True,
        scale=True, training=training, fused=True
    )


class AlexNet:
    def __init__(self, num_classes):
        # Must NHWC if run in c++ api.
        self.data_format = 'channels_last'
        self.num_classes = num_classes

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=tf.float32,
                             *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary.
         Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.
        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.

        """
        if dtype in (tf.float16,):
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def __call__(self, inputs, training):
        with tf.variable_scope('alexnet',
                               custom_getter=self._custom_dtype_getter):
            net = conv2d(inputs, 64, kernel_size=11, strides=4, padding=2)
            print('conv1', net)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, pool_size=3, strides=2)
            net = conv2d(net, 192, kernel_size=5, strides=4, padding=2)
            print('conv2', net)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, pool_size=3, strides=2)

            net = conv2d(net, 384, kernel_size=3, strides=1, padding=1)
            net = tf.nn.relu(net)
            net = conv2d(net, 256, kernel_size=3, strides=1, padding=1)
            net = tf.nn.relu(net)
            net = conv2d(net, 256, kernel_size=3, strides=1, padding=1)
            net = tf.nn.relu(net)
            # 256 x 6 x 6
            net = tf.layers.max_pooling2d(net, pool_size=3, strides=2)

            # classifier
            net = tf.layers.flatten(net)
            net = tf.layers.dropout(net, 0.5, training)
            net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
            net = tf.layers.dropout(net, 0.5, training)
            net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.num_classes)

            return net
