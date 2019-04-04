# Yhlin
import tensorflow as tf
import tensorflow.contrib as contrib
import preprocessing as P

import os
import random
import shutil
import json
import math
from collections import namedtuple

RecordDatasetInfo = namedtuple('RecordDatasetInfo',
                               ['train', 'val', 'link', 'num_classes',
                                'shuffle_buffer_size', 'num_train_files',
                                'num_val_files', 'name', 'label_file'])
ImageDatasetInfo = namedtuple('ImageDatasetInfo',
                              ['train', 'val', 'link', 'num_classes',
                               'shuffle_buffer_size', 'name'])

CUSTOM_DATASET_INFO = ImageDatasetInfo(train=1, val=1, link='', num_classes=2,
                                       shuffle_buffer_size=1, name='custom_dataset')
IMAGENET_CHANNEL_MEANS = [0.485, 0.456, 0.406]
IMAGENET_CHANNEL_STDS = [0.229, 0.224, 0.225]

CAR_INFO = {
    'train': 7370,
    'val': 818,
    'test': 8041,
    'dataset_name': 'Cars',
    'dataset_link': 'http://ai.stanford.edu/~jkrause/cars/car_dataset.html',
    'num_classes': 196
}

CAR_RECORD_INFO = RecordDatasetInfo(train=14567, val=1618,
                                    name='cars', num_classes=196,
                                    shuffle_buffer_size=5000,
                                    num_train_files=10,
                                    num_val_files=10,
                                    link='http://ai.stanford.edu/~jkrause/cars/car_dataset.html',
                                    label_file='')

RUBBISH_INFO = RecordDatasetInfo(train=2292, val=355, name='rubbish',
                                 num_classes=9, shuffle_buffer_size=1000,
                                 num_train_files=5, num_val_files=5, link='',
                                 label_file='')

TMP_INFO = {
    'train': 5743,
    'num_classes': 2
}


def get_filenames_labels(is_training, data_dir, save_class_to_idx=False,
                         save_dir='./'):
    """Images in data_dir are arranged in this way: ::

        data_dir/dog/xxx.png
        data_dir/dog/xxy.png
        data_dir/dog/xxz.png

        data_dir/cat/123.png
        data_dir/cat/nsdf3.png
        data_dir/cat/asd932_.png
    return filenames, labels
    """
    classes = [d for d in os.listdir(os.path.join(data_dir, 'train'))
               if os.path.isdir(os.path.join(data_dir, 'train', d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    if is_training:
        split_dir = os.path.join(data_dir, 'train')
    else:
        split_dir = os.path.join(data_dir, 'val')
    images = []
    labels = []
    for d in os.listdir(split_dir):
        for filename in os.listdir(os.path.join(split_dir, d)):
            if filename.endswith(('png', 'jpg', 'gif')):
                images.append(os.path.join(split_dir, d, filename))
                labels.append(class_to_idx[d])

    if save_class_to_idx:
        with open(os.path.join(save_dir, 'class_to_idx.json'), 'w',
                  encoding='utf-8') as fp:
            json.dump(class_to_idx, fp, indent=4)
    return images, labels


def split_dataset(val_ratio, data_dir, train_dir, val_dir):
    for d in os.listdir(data_dir):
        filenames = os.listdir(os.path.join(data_dir, d))

        num_samples = len(filenames)
        val_fnames = random.sample(filenames, int(num_samples * val_ratio))
        train_fnames = list(set(filenames) - set(val_fnames))
        train_class_dir = os.path.join(train_dir, d)
        val_class_dir = os.path.join(val_dir, d)
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)
        for val_fname in val_fnames:
            target_path = os.path.join(val_dir, d, val_fname)
            shutil.copyfile(
                os.path.join(data_dir, d, val_fname), target_path)
        for train_fname in train_fnames:
            target_path = os.path.join(train_dir, d, train_fname)
            shutil.copyfile(
                os.path.join(data_dir, d, train_fname), target_path)


# from scipy.io import loadmat
# # only train and val data
# def prepare_car_dataset(mat_path, data_dir=None, output_dir=None):
#     # Load label information
#     matrix = loadmat(mat_path)
#     print(matrix.keys())
#     # create as many directories as the number of classes in dataset (196)
#     print(matrix['annotations'].shape)
#     class_numbers = []  # 1 - 196
#     filename_class_map = {}
#     for i in range(matrix['annotations'].shape[1]):
#         filename = matrix['annotations'][0][i][5][0]
#         class_number = matrix['annotations'][0][i][4][0][0]
#         filename_class_map[filename] = class_number
#         class_numbers.append(class_number)
#
#     print(len(class_numbers))
#     print(max(class_numbers))
#     print(min(class_numbers))
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for i in range(min(class_numbers), max(class_numbers) + 1):
#         class_dir = os.path.join(output_dir, str(i))
#         if not os.path.exists(class_dir):
#             os.makedirs(os.path.join(output_dir, str(i)))
#
#     for filename in os.listdir(data_dir):
#         path = os.path.join(data_dir, filename)
#         target_path = os.path.join(output_dir,
#                                    str(filename_class_map[filename]),
#                                    filename)
#         shutil.copyfile(path, target_path)


def preprocess_car_image(image, is_training, fine_height,
                         fine_width, channels):
    if is_training:
        print('Read training data.')
    else:
        print('Read validation data.')
    if is_training:
        image = P.random_resized_crop(image, [224, 224])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.4)
    else:
        # image = tf.image.resize_images(image, [224, 224])
        crop_ratio = 0.875
        min_size = int(math.ceil(224 / crop_ratio))
        image = P.resize_keep_aspect(image, min_size)
        image = P.center_crop(image, 224, 224)

    image.set_shape([fine_height, fine_width, channels])
    # image = tf.image.per_image_standardization(image)
    image = P.normalize(image, IMAGENET_CHANNEL_MEANS, IMAGENET_CHANNEL_STDS, 3, tf.float32)
    return image


def preprocess_image_buffer(img_buffer, is_training, fine_height,
                            fine_width, channels):
    if is_training:
        print('Read training data.')
    else:
        print('Read validation data.')
    img = tf.image.decode_jpeg(img_buffer, channels=channels)
    if is_training:
        image = P.random_resized_crop(img, [224, 224])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.4)
    else:
        # image = tf.image.resize_images(image, [224, 224])
        crop_ratio = 0.875
        resize_min = int(math.ceil(224 / crop_ratio))
        image = P.resize_keep_aspect(img, resize_min)
        image = P.center_crop(image, 224, 224)

    image.set_shape([fine_height, fine_width, channels])
    # image = tf.image.per_image_standardization(image)
    image = P.normalize(image, IMAGENET_CHANNEL_MEANS, IMAGENET_CHANNEL_STDS, 3, tf.float32)
    return image


def preprocess_rubbish_image(image, is_training, fine_height,
                             fine_width, channels):
    if is_training:
        print('Read training data.')
    else:
        print('Read validation data.')
    if is_training:
        image = P.random_resized_crop(image, [224, 224])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)
        image = tf.image.random_hue(image, max_delta=0.4)
    else:
        # image = tf.image.resize_images(image, [224, 224])
        crop_ratio = 0.875
        min_size = int(math.ceil(224 / crop_ratio))
        image = P.resize_keep_aspect(image, min_size)
        image = P.center_crop(image, 224, 224)

    image.set_shape([fine_height, fine_width, channels])
    # image = tf.image.per_image_standardization(image)
    image = P.normalize(image, IMAGENET_CHANNEL_MEANS, IMAGENET_CHANNEL_STDS, 3, tf.float32)
    return image


def parse_image(filename, label, is_training, fine_height, fine_width, channels,
                preprocess_fn, dtype=tf.float32):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels)
    image = preprocess_fn(
        image, is_training, fine_height, fine_width, channels
    )
    # image = preprocess_image(image, 224, 224, is_training)
    image = tf.cast(image, dtype)
    return image, label


def parse_record(raw_record, is_training, fine_height, fine_width, channels,
                 preprocess_fn, dtype=tf.float32):
    def _parse_example_proto(example_serialized):
        """Parses an Example proto containing a training example of an image.
        The output of the build_image_data.py image preprocessing script is a
        dataset
        containing serialized Example protocol buffers. Each Example proto
        contains
        the following fields (values are included as examples):
      tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(image_data),
            'image/format': bytes_feature(image_format),
            'image/class/label': int64_feature(class_id),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
        }))
        Args:
          example_serialized: scalar Tensor tf.string containing a serialized
            Example protocol buffer.
        """
        # Dense features in Example proto.
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string,
                                                default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string,
                                               default_value='png'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        features = tf.parse_single_example(serialized=example_serialized,
                                           features=keys_to_features)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        return features['image/encoded'], label

    image_buffer, label = _parse_example_proto(raw_record)

    image = preprocess_fn(
        image_buffer, is_training, fine_height, fine_width, channels
    )
    image = tf.cast(image, dtype)

    return image, label


def get_cars_dataset_from_images(is_training, data_dir, batch_size, num_epochs=1,
                                 parse_fn=parse_image,
                                 preprocess_fn=preprocess_car_image,
                                 fine_height=224, fine_width=224,
                                 channels=3, dtype=tf.float32,
                                 save_class_to_idx=False,
                                 save_dir=None):
    filenames, labels = get_filenames_labels(is_training, data_dir,
                                             save_class_to_idx=save_class_to_idx,
                                             save_dir=save_dir)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.repeat(num_epochs)

    return process_image_dataset(dataset, is_training, batch_size,
                                 shuffle_buffer_size=CAR_INFO['train'],
                                 fine_height=fine_height,
                                 fine_width=fine_width,
                                 channels=channels,
                                 parse_fn=parse_fn,
                                 preprocess_fn=preprocess_fn,
                                 dtype=dtype)


def get_filenames(training, data_dir, record_dataset_info,
                  val_prefix='val'):
    if training:
        return [
            os.path.join(data_dir,
                         (record_dataset_info.name +
                          '_train_{:05d}-of-{:05d}.tfrecord'.format(
                              i,
                              record_dataset_info.num_train_files)))
            for i in range(record_dataset_info.num_train_files)]
    else:
        return [
            os.path.join(data_dir,
                         (record_dataset_info.name +
                          '_' + val_prefix + '_{:05d}-of-{:05d}.tfrecord'.format(
                              i,
                              record_dataset_info.num_val_files)))
            for i in
            range(record_dataset_info.num_val_files)]


def get_cars_dataset_from_record(is_training, data_dir, batch_size,
                                 num_epochs=1,
                                 parse_record_fn=parse_record,
                                 preprocess_fn=preprocess_image_buffer,
                                 fine_height=224, fine_width=224,
                                 channels=3, dtype=tf.float32,
                                 num_parallel_batches=1,
                                 val_prefix='val'):
    filenames = get_filenames(is_training, data_dir, CAR_RECORD_INFO,
                              val_prefix)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # cycle_length files will be read and deserialized in parallel.
    dataset = dataset.apply(contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10))

    return process_record_dataset(
        dataset, is_training, batch_size,
        shuffle_buffer_size=CAR_RECORD_INFO.shuffle_buffer_size,
        fine_height=fine_height,
        fine_width=fine_width,
        channels=channels,
        parse_record_fn=parse_record_fn,
        preprocess_fn=preprocess_fn,
        num_epochs=num_epochs,
        num_parallel_batches=num_parallel_batches,
        dtype=dtype)


def get_rubbish_dataset_from_record(is_training, data_dir, batch_size,
                                    num_epochs=1,
                                    parse_record_fn=parse_record,
                                    preprocess_fn=preprocess_image_buffer,
                                    fine_height=224, fine_width=224,
                                    channels=3, dtype=tf.float32,
                                    num_parallel_batches=1):
    filenames = get_filenames(is_training, data_dir, RUBBISH_INFO)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # cycle_length files will be read and deserialized in parallel.
    dataset = dataset.apply(contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=RUBBISH_INFO.num_train_files))

    return process_record_dataset(
        dataset, is_training, batch_size,
        shuffle_buffer_size=CAR_RECORD_INFO.shuffle_buffer_size,
        fine_height=fine_height,
        fine_width=fine_width,
        channels=channels,
        parse_record_fn=parse_record_fn,
        preprocess_fn=preprocess_fn,
        num_epochs=num_epochs,
        num_parallel_batches=num_parallel_batches,
        dtype=dtype)


def get_tmp_dataset(is_training, data_dir, batch_size, num_epochs=1,
                    parse_fn=parse_image,
                    preprocess_fn=preprocess_rubbish_image,
                    fine_height=224, fine_width=224,
                    channels=3, dtype=tf.float32,
                    save_class_to_idx=False,
                    save_dir=None):
    filenames, labels = get_filenames_labels(is_training, data_dir,
                                             save_class_to_idx=save_class_to_idx,
                                             save_dir=save_dir)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.repeat(num_epochs)

    return process_image_dataset(dataset, is_training, batch_size,
                                 shuffle_buffer_size=TMP_INFO['train'],
                                 fine_height=fine_height,
                                 fine_width=fine_width,
                                 channels=channels,
                                 parse_fn=parse_fn,
                                 preprocess_fn=preprocess_fn,
                                 dtype=dtype)


def process_image_dataset(dataset, is_training, batch_size,
                          shuffle_buffer_size,
                          parse_fn, preprocess_fn,
                          fine_height, fine_width, channels,
                          dtype=tf.float32):
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y: parse_fn(x, y, is_training, fine_height,
                              fine_width, channels, preprocess_fn, dtype),
        num_parallel_calls=4
    )

    # dataset = dataset.apply(contrib.data.batch_and_drop_remainder(
    # batch_size))

    dataset = dataset.prefetch(buffer_size=contrib.data.AUTOTUNE)
    return dataset


def process_record_dataset(dataset, is_training, batch_size,
                           shuffle_buffer_size,
                           parse_record_fn,
                           preprocess_fn,
                           fine_height, fine_width, channels,
                           num_epochs=1,
                           num_parallel_batches=1,
                           dtype=tf.float32):
    # Prefetches a batch at a time to smooth out the time taken to load
    # input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffles records before repeating to respect epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)

    #
    # dataset = dataset.map(
    #     lambda x, y: parse_fn(x, y, is_training, fine_height,
    #                           fine_width, channels, preprocess_fn, dtype),
    #     num_parallel_calls=4
    # )
    dataset = dataset.apply(contrib.data.map_and_batch(
        lambda v: parse_record_fn(v, is_training, fine_height,
                                  fine_width, channels, preprocess_fn,
                                  dtype),
        batch_size,
        num_parallel_batches=num_parallel_batches,
        drop_remainder=False))

    # dataset = dataset.batch(batch_size)

    # dataset = dataset.apply(contrib.data.batch_and_drop_remainder(
    # batch_size))

    dataset = dataset.prefetch(buffer_size=contrib.data.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    split_dataset(0.1, '...output',
                  '...output_train',
                  '...output_val')
