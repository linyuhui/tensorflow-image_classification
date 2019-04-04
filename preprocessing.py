import tensorflow as tf
import random
import math


def center_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
      image: a 3-D image tensor
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      3-D tensor with cropped image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def normalize(image, means, stds, num_channels, dtype=tf.float32):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.
      num_channels: number of color channels in the image that will be
      distorted.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    image = tf.cast(image, dtype)
    image = image / 255.0

    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    if len(stds) != num_channels:
        raise ValueError('len(stds) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)
    stds = tf.expand_dims(tf.expand_dims(stds, 0), 0)
    return (image - means) / stds


def random_resized_crop(img, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    """size: (height, width)"""
    shape = tf.shape(input=img)
    img_h, img_w = shape[0], shape[1]
    # print(img_h, img_w)

    index = tf.constant(0, dtype=tf.int32)
    ready = tf.constant(False)

    # Fallback
    crop_w = tf.minimum(img_h, img_w)
    crop_h = tf.identity(crop_w)
    crop_top = (img_h - crop_w) // 2
    crop_left = (img_w - crop_w) // 2
    keep_attempt = lambda i, r, top, left, height, width: tf.logical_and(
        tf.less(i, 10), tf.logical_not(r))

    def attempt(i, _ready, _crop_top, _crop_left, _crop_h, _crop_w):
        area = tf.cast(img_h * img_w, tf.float32)
        target_area = tf.random_uniform([], *scale) * area
        aspect_ratio = tf.random_uniform([], *ratio)

        w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
        h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)),
                    tf.int32)

        w, h = tf.cond(tf.less(tf.random_uniform([]), 0.5),
                       lambda: (w, h), lambda: (h, w))

        _ready = tf.cond(tf.logical_and(tf.less_equal(w, img_w),
                                        tf.less_equal(h, img_h)),
                         lambda: tf.logical_not(_ready),
                         lambda: _ready)

        _crop_top, _crop_left, _crop_h, _crop_w = tf.cond(
            _ready,
            lambda:
            (tf.random_uniform([], 0, img_h - h + 1, dtype=tf.int32),
             tf.random_uniform([], 0, img_w - w + 1, dtype=tf.int32),
             h, w),
            lambda: (_crop_top, _crop_left, _crop_h, _crop_w)
        )

        # return tf.Print(i + 1, [i]), tf.Print(_ready, [_ready], 'ready'),
        # _crop_top, _crop_left, _crop_h, _crop_w
        return i + 1, _ready, _crop_top, _crop_left, _crop_h, _crop_w

    _, ready, crop_top, crop_left, crop_h, crop_w = tf.while_loop(
        cond=keep_attempt,
        body=attempt,
        loop_vars=[index, ready, crop_top, crop_left, crop_h, crop_w]
    )

    # return ready, crop_top, crop_left, crop_h, crop_w
    # crop
    img = tf.slice(
        img, [crop_top, crop_left, 0], [crop_h, crop_w, -1])
    # resize
    return tf.image.resize_images(img, size)


def _resize_image(img, height, width):
    """Simple wrapper around tf.resize_images.
    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.
    Args:
      image: A 3-D image `Tensor`.
      height: The target height for the resized image.
      width: The target width for the resized image.
    Returns:
      resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    return tf.image.resize_images(
        img, [height, width], method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)


def resize_keep_aspect(img, resize_min):
    shape = tf.shape(input=img)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(img, new_height, new_width)


def _smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


if __name__ == '__main__':
    img = tf.zeros([640, 480, 1])
    crop_ratio = 0.875
    min_size = int(math.ceil(224 / crop_ratio))
    res = resize_keep_aspect(img, min_size)
    with tf.Session() as sess:
        print(sess.run(tf.shape(res)))
