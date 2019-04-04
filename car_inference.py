import tensorflow as tf
import os
from mobilenet import MobileNet
from dataset_factory import CAR_RECORD_INFO
from dataset_utils import read_label_file
import preprocessing_on_opencv_image as P

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_model():
    input = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input')
    model = MobileNet(num_classes=CAR_RECORD_INFO.num_classes)
    logit = model(input, False)
    pred_class_id = tf.cast(tf.argmax(logit, axis=1), tf.int32)
    pred_prob = tf.nn.softmax(logit)
    return input, [pred_class_id, pred_prob]


def main(ckpt, img_path, label_path):
    input, output = build_model()
    preprocess_fn = P.preprocess
    img = preprocess_fn(img_path)
    saver = tf.train.Saver()
    id_to_class = read_label_file(os.path.dirname(label_path),
                                  os.path.basename(label_path))
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output_value = sess.run(output, feed_dict={input: img})
        pred_class_id = output_value[0]
        pred_prob = output_value[1]
        print('prediction id:', pred_class_id[0])
        print('prediction class name : ', id_to_class[pred_class_id[0]])
        print('probability:')
        print(pred_prob[0][pred_class_id[0]])


if __name__ == '__main__':
    main('.../car_ckpts/model-70146',
         '.../015851.jpg',
         '.../datasets/cars/labels.txt')

