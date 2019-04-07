import tensorflow as tf


def get_model_1(model_in, encoder_lname, decoder_lname):
    tf_out = tf.layers.Conv2D(16, 4, 2, activation=tf.nn.leaky_relu, padding='valid', name='Conv_1')(model_in)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2D(32, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Conv_2')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2D(32, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Conv_3')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2D(32, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Conv_4')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Conv_5')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2D(64, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Conv_6')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2D(64, 3, strides=1,
                              activation=tf.nn.leaky_relu, padding='valid', name=encoder_lname)(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2DTranspose(64, 3, strides=1,
                                       activation=tf.nn.leaky_relu, padding='valid', name='Deconv_1')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2DTranspose(64, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Deconv_2')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2DTranspose(64, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Deconv_3')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2DTranspose(32, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Deconv_4')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2DTranspose(32, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Deconv_5')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2DTranspose(32, 3, 1, activation=tf.nn.leaky_relu, padding='valid', name='Deconv_6')(tf_out)
    print(tf_out.shape)
    tf_out = tf.layers.Conv2DTranspose(1, 4, 2, activation=tf.nn.leaky_relu, padding='valid', name=decoder_lname)(tf_out)
    print(tf_out.shape)
    return tf_out
