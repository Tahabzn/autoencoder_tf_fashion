import tensorflow as tf
import re


def get_model_1(model_in, prediction_lname, load_auto_path):
    current_graph = tf.get_default_graph()
    temp_graph = tf.Graph()
    auto_in = tf.contrib.copy_graph.copy_op_to_graph(model_in, temp_graph, [])
    with temp_graph.as_default():
        tf.train.import_meta_graph(load_auto_path, input_map={'IteratorGetNext:0': auto_in})
        needed_ops = ['Conv_', 'Encoder']
        varlist = []
        for each_var in temp_graph.get_collection('variables'):
            for each_str in needed_ops:
                if each_var.name.startswith(each_str):
                    if re.match('^((?!Train_op).)*$', each_var.name):
                        new_var = tf.contrib.copy_graph.copy_variable_to_graph(each_var, current_graph)
                        varlist.append(new_var)
        for each_op in tf.get_default_graph().get_operations():
            for each_str in needed_ops:
                if each_op.name.startswith(each_str):
                    if re.match('^((?!Train_op).)*$', each_op.name):
                        _ = tf.contrib.copy_graph.copy_op_to_graph(each_op, current_graph, [])
        print('Enocder graph loaded from {}'.format(load_auto_path))
    encoder = tf.get_default_graph().get_tensor_by_name('Encoder/LeakyRelu:0')
    encoder_shape = temp_graph.get_tensor_by_name('Encoder/LeakyRelu:0').get_shape().as_list()
    encoder_shape[0] = -1
    encoder = tf.reshape(encoder, encoder_shape)  # workaround since the shapes are not defined when copying the variables
    print('Encoder shape: {}'.format(encoder.shape))
    tf_out = tf.layers.Flatten()(encoder)
    print('{}\'s shape {}'.format(tf_out.name, tf_out.shape))
    tf_out = tf.layers.Dense(128, activation=tf.nn.leaky_relu, name='Dense_1')(tf_out)
    print('{}\'s shape {}'.format(tf_out.name, tf_out.shape))
    tf_out = tf.layers.Dense(10, activation=tf.identity, name=prediction_lname)(tf_out)
    print('{}\'s shape {}'.format(tf_out.name, tf_out.shape))
    return tf_out, varlist

