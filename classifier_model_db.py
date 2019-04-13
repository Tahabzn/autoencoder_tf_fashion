import tensorflow as tf
import re
import os


def get_model_1(model_in, prediction_lname, load_auto_path):
    current_graph = tf.get_default_graph()
    temp_graph = tf.Graph()
    auto_in = tf.contrib.copy_graph.copy_op_to_graph(model_in, temp_graph, [])
    with temp_graph.as_default():
        tf.train.import_meta_graph(load_auto_path, input_map={'IteratorGetNext:0': auto_in})
        needed_ops = ['Conv_', 'Encoder', 'IteratorGetNext']
        varlist = []
        for each_var in temp_graph.get_collection('trainable_variables'):
            for each_str in needed_ops:
                if each_var.name.startswith(each_str):
                    # new_var = tf.contrib.copy_graph.copy_variable_to_graph(each_var, current_graph)
                    varlist.append(each_var)
        auto_init_op, auto_init_feed_dict = tf.contrib.framework.assign_from_checkpoint(os.path.splitext(load_auto_path)[0], varlist)
        auto_init_op = tf.contrib.copy_graph.copy_op_to_graph(auto_init_op, current_graph, [])
        auto_init_feed_dict_new = {}
        for k, v in auto_init_feed_dict.items():
            k_new = tf.contrib.copy_graph.copy_op_to_graph(k, current_graph, [])
            auto_init_feed_dict_new[k_new] = v
        for each_op in tf.get_default_graph().get_operations():
            for each_str in needed_ops:
                if each_op.name.startswith(each_str):
                    if re.match('^((?!Train_op).)*$', each_op.name):
                        _ = tf.contrib.copy_graph.copy_op_to_graph(each_op, current_graph, [])
    encoder = tf.get_default_graph().get_tensor_by_name('Encoder/LeakyRelu:0')
    print('Encoder shape: {}'.format(encoder.shape))
    tf_out = tf.layers.Flatten()(encoder)
    print(tf_out.shape)
    tf_out = tf.layers.Dense(128, activation=tf.nn.leaky_relu, name='Dense_1')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Dense(10, activation=tf.identity, name=prediction_lname)(tf_out)
    print(tf_out.shape)
    return tf_out, auto_init_op, auto_init_feed_dict_new

