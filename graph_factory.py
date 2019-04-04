import tensorflow as tf
import os

from tensorflow.python.framework.graph_util import convert_variables_to_constants
import numpy as np
from mobilenet import MobileNet
from dataset_factory import CAR_RECORD_INFO
import argparse
import preprocessing_on_opencv_image as P 

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='./ckpts/model')
parser.add_argument('--output_tensor_names', type=str, default='class_id,prob')
parser.add_argument('--input_tensor_name', type=str, default='input')
parser.add_argument('--graph_def_file', default='./graph_def.pb')
parser.add_argument('--operation', choices=['freeze', 'load'], default=None)
parser.add_argument('--image_path', default='./gu.jpg')
args = parser.parse_args()


def build_model():
    model = MobileNet(num_classes=CAR_RECORD_INFO.num_classes)
    inputs = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input')
    logits = model(inputs, training=False)
    classe_preds = tf.cast(tf.argmax(input=logits, axis=1), tf.int32, name='class_id')
    prob_preds = tf.nn.softmax(logits, name='prob')
    return inputs, [classe_preds, prob_preds]


def freeze_graph(inputs, args):
    # only need node name in graph.
    input, predictions = build_model()
    print('Input tensor:', input)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, args.ckpt)
        sess.run(predictions, feed_dict={input: inputs})
        if args.output_tensor_names is None:
            output_node_names = [
                name[:-2] for name in args.output_tensor_names.split(',')]
        else:
            output_node_names = []
            if isinstance(predictions, list):
                for p in predictions:
                    output_node_names += [p.name[:-2]]
        try:
            # It is a NoOp
            sess.graph.get_operation_by_name('init_all_tables')
            output_node_names.append(
                'init_all_tables')  # If you need to run table initializer.
        except KeyError:
            pass

        print('Output tensor: ', predictions)
        graph_def = convert_variables_to_constants(sess, sess.graph_def,
                                                   output_node_names)

        graph_def_name = os.path.basename(args.graph_def_file)
        graph_dir = os.path.dirname(args.graph_def_file)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        print('Save to :', args.graph_def_file)
        tf.train.write_graph(graph_def, graph_dir, graph_def_name, as_text=False)
        print('Done.')


def load_frozen_graph(args):
    with open(args.graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    preprocess_fn = P.preprocess
    img = preprocess_fn(args.image_path)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=graph) as sess:
            try:
                print(graph.get_operation_by_name('init_all_tables'))
                sess.run(graph.get_operation_by_name('init_all_tables'))
            except KeyError:
                print('table initialization is not required.')
                pass
            result = sess.run(args.output_tensor_names.split(','),
                              feed_dict={args.input_tensor_name: img})
            print(result)


def main(args):
    print('tensorflow version: ', tf.__version__)
    if args.operation == 'freeze':
        inputs = np.random.randn(1, 224, 224, 3)
        freeze_graph(inputs, args)
    elif args.operation == 'load':
        load_frozen_graph(args)
    else:
        raise TypeError


if __name__ == '__main__':
    main(args)
