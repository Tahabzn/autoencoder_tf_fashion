from datetime import datetime
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import progressbar
import cv2
import tensorflow as tf
import fashion_mnist_utils
import classifier_model_db
from scipy.special import softmax

import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='store_true', help='verbose flag')
subparsers = parser.add_subparsers(help='commands', dest='mode')
# train command
train_p = subparsers.add_parser('train', help='to train the model')
subparsers_auto_p = train_p.add_subparsers(help='commands', dest='training_mode')
cont_p = subparsers_auto_p.add_parser('continue', help=' to continue training')
cont_p.add_argument('--model', '-m', type=str,
                    help='path to pre-trained model (.meta file) to continue training', required=True)
auto_p = subparsers_auto_p.add_parser('start', help=' to start a new training')
auto_p.add_argument('-a', '--autoencoder', type=str,
                    help='path to pre-trained autoencoder (.meta file)', required=True)
auto_p.add_argument('--init-random', '-r', action='store_true', help='initialize random weights to encoder (.meta file)'
                                                                     'instead of pre-trained weights')
# test command
test_p = subparsers.add_parser('test', help='to test the model')
test_p.add_argument('--model', '-m', type=str, help='path to pre-trained model (.meta file)', required=True)
args = parser.parse_args()


def save_model_epoch(comment):
    filename, ext = os.path.splitext(out_file_name)
    model_filename = filename + '_' + comment
    saver.save(sess, os.path.join(out_path, model_filename))


def display_prediction(img, label, prediction):
    label_dict = fashion_mnist_utils.get_label_dict()
    plt.title('Image lable ={} pred ={}'.format(label_dict[label], label_dict[prediction]))
    plt.imshow(img)
    plt.show()


# Training Parameters
mode = args.mode
max_epochs = 50
batchsize = 64
learning_rate = 0.001
if args.mode == 'train':
    if args.training_mode == 'continue':
        continue_from_checkpoint = True
        load_model_path = args.model  # './out/classifier/sample/classifier_latest.meta'
    else:
        continue_from_checkpoint = False
        load_auto_path = args.autoencoder  # './out/autoencoder/sample/autoencoder_best.meta'
        init_auto_weights_rand = args.init_random
else:
    continue_from_checkpoint = True
    load_model_path = args.model  # './out/classifier/sample/classifier_best.meta'

final_model_path = './out/classifier/classifier.meta'

# Dataset Preparation
train_images = fashion_mnist_utils.extract_images('./data/train-images-idx3-ubyte.gz', 60000)
test_images = fashion_mnist_utils.extract_images('./data/t10k-images-idx3-ubyte.gz', 10000)
train_labels = fashion_mnist_utils.extract_labels('./data/train-labels-idx1-ubyte.gz', 60000)
train_labels_hot = tf.keras.utils.to_categorical(train_labels)
test_labels = fashion_mnist_utils.extract_labels('./data/t10k-labels-idx1-ubyte.gz', 10000)
test_labels_hot = tf.keras.utils.to_categorical(test_labels)
train_X, val_X, train_Y, val_Y = train_test_split(train_images, train_labels_hot, test_size=0.2, random_state=13)
# Shapes of training set
print("Training images shape: {shape}".format(shape=train_X.shape))
print("Validation images shape: {shape}".format(shape=val_X.shape))
print("Test images shape: {shape}".format(shape=test_images.shape))

# Out folder creation
out_path, out_file_name = os.path.split(final_model_path)
if continue_from_checkpoint:
    out_path = os.path.join(out_path, load_model_path.split(os.sep)[-2])
    out_path_train_tb = os.path.join(out_path, 'train_log')
    out_path_val_tb = os.path.join(out_path, 'val_log')
    print('Out folder {} is used for logging & model saving'.format(out_path))
else:
    out_path = os.path.join(out_path, 'Training_' + str(datetime.now().strftime('_%Y%m%d_%H%M%S')))
    out_path_train_tb = os.path.join(out_path, 'train_log')
    out_path_val_tb = os.path.join(out_path, 'val_log')
    os.makedirs(out_path_train_tb)
    os.makedirs(out_path_val_tb)
    print('Out folder {} created'.format(out_path))

# Model definition
tf.set_random_seed(12)
if not continue_from_checkpoint:
    model_in = tf.placeholder(dtype=tf.float32, shape=(None,) + train_X.shape[1:], name='Model_in')
    model_out = tf.placeholder(dtype=tf.float32, shape=(None,) + train_Y.shape[1:], name='Model_out')
    dataset = tf.data.Dataset.from_tensor_slices((model_in, model_out))
    dataset = dataset.shuffle(buffer_size=train_X.shape[0]).batch(batch_size=batchsize)
    data_iter = dataset.make_initializable_iterator()
    data_iter_init = data_iter.make_initializer(dataset, name='Data_itr_init')
    next_batch = data_iter.get_next()
    img_classifier, varlist = classifier_model_db.get_model_1(next_batch[0], 'Predictor', load_auto_path)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_batch[1],
                                                                     logits=img_classifier), name='Loss')
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Train_op').minimize(loss=loss)
    # Tensorboard variables initialization
    with tf.name_scope('performance'):
        tf_itr_loss_ph = tf.placeholder(dtype=tf.float32, shape=None, name='Tf_itr_loss_ph')
        tf_itr_loss_summary = tf.summary.scalar('Loss_itr', tf_itr_loss_ph)
        tf_epoch_loss_ph = tf.placeholder(dtype=tf.float32, shape=None, name='Tf_epoch_loss_ph')
        tf_epoch_loss_summary = tf.summary.scalar('Loss_epoch', tf_epoch_loss_ph)
    print('Model newly created')
else:
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph(load_model_path)
    print('Model loaded from checkpoint ' + load_model_path)

# Model training
saver = tf.train.Saver()
if not continue_from_checkpoint:
    training_loss = []
    val_loss = []
    # tensorboard init
    itr_count_tb = 0
    each_epoch = 0
else:
    training_loss, val_loss = np.load(os.path.join(out_path, 'loss.npy'))
    training_loss = training_loss.tolist()
    val_loss = val_loss.tolist()
    # tensorboard init
    itr_count_tb, each_epoch = np.load(os.path.join(out_path, 'tb.npy'))
    itr_count_tb += 1
    each_epoch += 1

sess = tf.Session()
if not continue_from_checkpoint:
    sess.run(tf.global_variables_initializer())
    # loading pretrained weights of encoder
    if not init_auto_weights_rand:
        auto_saver = tf.train.Saver(varlist)
        auto_saver.restore(sess, os.path.splitext(load_auto_path)[0])
        print('Encoder weights are loaded from {}'.format(load_auto_path))
    else:
        print('Encoder weights are randomly initialized')
else:
    imported_meta.restore(sess, os.path.splitext(load_model_path)[0])

if mode == 'train':
    # check max epochs for continue training
    if each_epoch >= max_epochs:
        print('Max epoch count already reached! please change the max limit and retry')
        assert each_epoch < max_epochs

    # Tensorboard writer
    summ_writer_train = tf.summary.FileWriter(out_path_train_tb, graph=sess.graph)
    summ_writer_val = tf.summary.FileWriter(out_path_val_tb, graph=sess.graph)
    # Training
    while each_epoch < max_epochs:
        start = datetime.now()
        iteration = 0
        train_loss_epoch = 0
        max_teration = math.ceil(train_X.shape[0] / batchsize)
        # Progressbar
        bar = progressbar.ProgressBar(maxval=max_teration+1,
                                      widgets=[progressbar.Bar('=', 'Epoch {} Training:['.format(each_epoch+1), ']'),
                                               ' ', progressbar.Percentage()])
        bar.start()
        if args.verbose:
            print('Epoch : {}'.format(each_epoch + 1))
            print('Training...')
        sess.run('Data_itr_init', feed_dict={"Model_in:0": train_X, "Model_out:0": train_Y})
        while iteration < max_teration:
            _, iteration_loss = sess.run(['Train_op', 'Loss:0'])
            iteration += 1
            itr_count_tb += 1
            # Tensorboard update
            summ_itr = sess.run('performance/Loss_itr:0',
                                feed_dict={'performance/Tf_itr_loss_ph:0': iteration_loss})
            summ_writer_train.add_summary(summ_itr, itr_count_tb)
            train_loss_epoch += iteration_loss
            bar.update(iteration)
            if args.verbose:
                if iteration % max(1, (max_teration//10)) == 0:
                    print('iteration {} loss: {:0.4f}'.format(iteration, iteration_loss))
        train_loss_epoch = train_loss_epoch / iteration
        # Tensorboard update
        summ = sess.run('performance/Loss_epoch:0',
                        feed_dict={'performance/Tf_epoch_loss_ph:0': train_loss_epoch})
        summ_writer_train.add_summary(summ, each_epoch + 1)
        summ_writer_train.flush()

        # Validation
        if args.verbose:
            print('Validating...')
        iteration_val = 0
        val_loss_epoch = 0
        max_teration = math.ceil(val_X.shape[0] / batchsize)
        # Progressbar
        bar = progressbar.ProgressBar(maxval=max_teration+1,
                                      widgets=[progressbar.Bar('=', 'Epoch {} Validating:['.format(each_epoch+1), ']'),
                                               ' ', progressbar.Percentage()])
        sess.run('Data_itr_init', feed_dict={"Model_in:0": val_X, "Model_out:0": val_Y})
        while iteration_val < max_teration:
            iteration_loss = sess.run('Loss:0')
            iteration_val += 1
            val_loss_epoch += iteration_loss
            bar.update(iteration_val)
        val_loss_epoch = val_loss_epoch / iteration_val

        # Tensorboard update
        summ = sess.run('performance/Loss_epoch:0',
                        feed_dict={'performance/Tf_epoch_loss_ph:0': val_loss_epoch})
        summ_writer_val.add_summary(summ, each_epoch + 1)
        summ_writer_val.flush()

        if args.verbose:
            print('-' * 100)
            print('Training   : Epoch {0} has loss {1:0.6f} & metric {1:0.6f}'.format(each_epoch + 1, train_loss_epoch))
            print('Validation : Epoch {0} has loss {1:0.6f} & metric {1:0.6f}'.format(each_epoch + 1, val_loss_epoch))
            print('-' * 100)

        # Model checkpoint
        if len(val_loss) != 0 and val_loss_epoch < min(val_loss):
            if args.verbose:
                print('Saving the best model...')
            save_model_epoch('best')

        training_loss.append(train_loss_epoch)
        val_loss.append(val_loss_epoch)

        # saving the latest model
        save_model_epoch('latest')
        np.save(os.path.join(out_path, 'tb'), [itr_count_tb, each_epoch])
        np.save(os.path.join(out_path, 'loss'), [training_loss, val_loss])

        if args.verbose:
            print('Time taken = {}'.format((datetime.now() - start).total_seconds()))

        each_epoch += 1

if mode == 'test':
    sess.run('Data_itr_init', feed_dict={"Model_in:0": val_X, "Model_out:0": val_Y})
    in_img, label, pred = sess.run(['IteratorGetNext:0', 'IteratorGetNext:1', 'Predictor/Identity:0'])
    sample_no = np.random.randint(0, batchsize - 1, 1)
    orig_sample = np.squeeze(in_img[sample_no, :, :, :], axis=0)
    orig_sample = cv2.cvtColor(orig_sample, cv2.COLOR_GRAY2RGB)
    label_sample = label[sample_no]
    pred_sample = pred[sample_no]
    display_prediction(orig_sample, np.argmax(softmax(label_sample)), np.argmax(softmax(pred_sample)))
sess.close()