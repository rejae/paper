# coding=utf-8
import os
import sys
import getopt
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph


argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "m:v:r:", ["modelname=","version=","rootdir="])
except getopt.GetoptError:
    print('parameter error')
    sys.exit(2)
for opt, arg in opts:
    if opt in ['-m', '--modelname']:
        modelname = arg
    elif opt in ['-v', '--version']:
        version = arg
    elif opt in ['-r', '--rootdir']:
        rootdir = arg
print('model:%s' % modelname)
print('version:%s' % version)
print('rootdir:%s' % rootdir)


def get_graph_def_from_saved_model(saved_model_dir):
    print("Load graph from model: {}".format(saved_model_dir))
    with tf.Session() as session:
        meta_graph_def = tf.saved_model.loader.load(
            session,
            tags=[tag_constants.SERVING],
            export_dir=saved_model_dir
        )
        return meta_graph_def.graph_def


def get_graph_def_from_file(graph_filepath):
    print("Load graph from file: {}".format(graph_filepath))
    with tf.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


def describe_graph(graph_def, show_nodes=False):
    print('Input Feature Nodes: {}'.format(
        [node.name for node in graph_def.node if node.op == 'Placeholder']))
    print('')
    print('Unused Nodes: {}'.format(
        [node.name for node in graph_def.node if 'unused' in node.name]))
    print('')
    print('Output Nodes: {}'.format(
        [node.name for node in graph_def.node if (
                'predictions' in node.name or 'Softmax' in node.name)]))
    print('')
    print('Quantization Nodes: {}'.format(
        [node.name for node in graph_def.node if 'quant' in node.name]))
    print('')
    print('Constant Count: {}'.format(
        len([node for node in graph_def.node if node.op == 'Const'])))
    print('')
    print('Variable Count: {}'.format(
        len([node for node in graph_def.node if 'Variable' in node.op])))
    print('')
    print('Identity Count: {}'.format(
        len([node for node in graph_def.node if node.op == 'Identity'])))
    print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

    if show_nodes == True:
        for node in graph_def.node:
            print('Op:{} - Name: {}'.format(node.op, node.name))
    print()


def freeze_model(saved_model_dir, output_node_names, output_graph_filename):
    initializer_nodes = ''
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags=tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )
    print('graph freezed!')


def optimize_graph(model_dir, graph_filename, transforms, output_graph_filename):
    input_names = ['serving_input_ids', 'serving_input_mask', 'serving_segment_ids']
    output_names = ['loss/Softmax']
    if graph_filename is None:
        graph_def = get_graph_def_from_saved_model(model_dir)
    else:
        graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
    optimized_graph_def = TransformGraph(
        graph_def,
        input_names,
        output_names,
        transforms)
    tf.train.write_graph(optimized_graph_def,
                         logdir=model_dir,
                         as_text=False,
                         name=output_graph_filename)
    print('Graph optimized!')


def convert_graph_def_to_saved_model(export_dir, graph_filepath):
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
    graph_def = get_graph_def_from_file(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={'input_ids': session.graph.get_tensor_by_name('serving_input_ids:0'),
                    'input_mask': session.graph.get_tensor_by_name('serving_input_mask:0'),
                    'segment_ids': session.graph.get_tensor_by_name('serving_segment_ids:0')},
            outputs={'output': session.graph.get_tensor_by_name('loss/Softmax:0')}
    )
    print('Optimized graph converted to SavedModel!')


model_root = os.path.join(rootdir, 'middle', 'all', 'model')
saved_model_dir = [f.path for f in os.scandir(model_root) if f.is_dir()][0]
optimize_dir = os.path.join(rootdir, 'middle', 'optimize')
os.mkdir(optimize_dir)

frozen_filepath = os.path.join(optimize_dir, 'frozen_graph.pb')
freeze_model(saved_model_dir, "loss/Softmax", frozen_filepath)
describe_graph(get_graph_def_from_file(frozen_filepath))

transforms = [
 'remove_nodes(op=Identity)',
 'merge_duplicate_nodes',
 'strip_unused_nodes',
 'fold_constants(ignore_errors=true)',
 'fold_batch_norms',
]
optimize_filepath = os.path.join(optimize_dir, 'optimized_graph.pb')
optimize_graph(optimize_dir, 'frozen_graph.pb', transforms, optimize_filepath)

#  'quantize_nodes' will case error when predict, remove it from optimize
transforms = [
 'quantize_weights',
]
quantization_filepath = os.path.join(optimize_dir, 'quantization_graph.pb')
optimize_graph(optimize_dir, "optimized_graph.pb", transforms, quantization_filepath)

describe_graph(get_graph_def_from_file(quantization_filepath))
output_dir = os.path.join(rootdir, "model")
convert_graph_def_to_saved_model(output_dir, quantization_filepath)
