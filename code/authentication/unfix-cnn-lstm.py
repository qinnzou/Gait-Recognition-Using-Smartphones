"""Human activity recognition using smartphones dataset and an LSTM RNN."""

# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

# The MIT License (MIT)
#
# Copyright (c) 2016 Guillaume Chevalier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Also thanks to Zhao Yu for converting the ".ipynb" notebook to this ".py"
# file which I continued to maintain.

# Note that the dataset must be already downloaded for this script to work.
# To download the dataset, do:
#     $ cd data/
#     $ python download_dataset.py


import tensorflow as tf
import os
import numpy as np


# Load "X" (the neural network's training and testing inputs)

def changex(xchanged):
    X_signals = []
    for l in xchanged:
        arr=l.flatten()
        X_signals.append(arr)
    return X_signals
def load_X(path):
    X_signals = []
    files = os.listdir(path)
    for my_file in files:
        fileName = os.path.join(path,my_file)
        file = open(fileName, 'r')
        X_signals.append(
            [np.array(cell, dtype=np.float32) for cell in [
                row.strip().split(' ') for row in file
            ]]
        )
        file.close()
        #X_signals = 6*totalStepNum*128
    X_signals = np.transpose(np.array(X_signals), (1, 0, 2))#(totalStepNum*6*128)
    return X_signals.reshape(-1,6,256,1)#(totalStepNum*6*128*1)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    y_ = y_ - 1
    #one_hot
    y_ = y_.reshape(len(y_))
    n_values = 2
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.n_layers = 1   # nb of layers
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = 32 # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 512

        # LSTM structure
        self.n_inputs = 384  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = 2  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(_X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells

    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".

    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.

      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2]*config.n_layers, state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


def one_hot(y_):
    """
    Function to encode output labels from number indexes.

    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


if __name__ == "__main__":

    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "_acc_x",
        "_acc_y",
        "_acc_z",
        "_gyr_x",
        "_gyr_y",
        "_gyr_z",
    ]

    DATA_PATH = "data/"
    DATASET_PATH = "data/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    TRAIN = "a/"
    TEST = "b/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "data/"  + "train" + signal + '.txt' for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "data/" + "test" + signal + '.txt' for signal in INPUT_SIGNAL_TYPES
    ]


    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    X_train = load_X('./data/a/data/')
    X_test = load_X('./data/b/data/')

    train_label = load_y(y_train_path)
    test_label = load_y(y_test_path)

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, test_label.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Let's get serious and build the neural network
    # ------------------------------------------------------
# =============================================================================
# 
#     X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
#     Y = tf.placeholder(tf.float32, [None, config.n_classes])
#     batch_size = 512
# =============================================================================
    X_ = tf.placeholder(tf.float32, [None, 6, 128, 1],name='cnn_X')
    X2 = tf.placeholder(tf.float32, [None, 6, 128, 1],name='cnn_X2')
    label_ = tf.placeholder(tf.float32, [None, 2],name='cnn_Y')
    
    #input shape [batch, in_height, in_width, in_channels]
    #kernel shape [filter_height, filter_width, in_channels, out_channels]
    '''
    	1*9
    	stride = 2 
    	padding
    	6*128->6*64*32
    '''
    W_conv1 = weight_variable([1, 9, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(X_, W_conv1, strides=[1, 1, 2, 1], padding='SAME') + b_conv1)
    h_conv12 = tf.nn.relu(
        tf.nn.conv2d(X2, W_conv1, strides=[1, 1, 2, 1], padding='SAME') + b_conv1)    
    '''
    	pooling
    	6*64*32->6*32*32
    '''
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
    h_pool12 = tf.nn.max_pool(h_conv12, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
    '''
    	1*3
    	stride = 1
    	6*32*32->6*32*64
    '''
    W_conv2 = weight_variable([1, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_conv22 = tf.nn.relu(
        tf.nn.conv2d(h_pool12, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    '''
    	1*3
    	stride = 1
    	padding
    	6*32*64->6*32*128
    '''
    
    W_conv3 = weight_variable([1, 3, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(
        tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    h_conv32 = tf.nn.relu(
        tf.nn.conv2d(h_conv22, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    
    '''
    '''
    '''
    	pooling
    	6*32*128->6*16*128
    '''
    h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],padding='VALID')
    h_pool22 = tf.nn.max_pool(h_conv32, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],padding='VALID')
    '''
    	6*1
    	6*32*128->1*16*128
    '''
    W_conv4 = weight_variable([6, 1, 128, 128])
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(
        tf.nn.conv2d(h_pool2, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
    h_conv42 = tf.nn.relu(
        tf.nn.conv2d(h_pool22, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
    t1 = tf.reshape(h_conv4, [-1, 16, 128]) 
    t2 = tf.reshape(h_conv42, [-1, 16, 128])
    ct = tf.concat([t1,t2],1)     # -
    pred_Y = LSTM_Network(ct, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=label_, logits=pred_Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(label_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # --------------------------------------------
    # Step 4: Hooray, now train the neural network
    # --------------------------------------------

    # Note that log_device_placement can be turned ON but will cause console spam with RNNs.
    saver = tf.train.Saver(max_to_keep=1)

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X_: X_train[start:end,:,:128],
                                           X2:X_train[start:end,:,128:],
                                           label_: train_label[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, cost], 
            feed_dict={
                X_: X_test[:,:,:128],
                X2: X_test[:,:,128:],
                label_: test_label
            }
        )
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        if accuracy_out>best_accuracy:
            saver.save(sess, './lstm_ckpt/model')
            best_accuracy = accuracy_out


    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    # ------------------------------------------------------------------
    # Step 5: Training is good, but having visual insight is even better
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook

    # ------------------------------------------------------------------
    # Step 6: And finally, the multi-class confusion matrix and metrics!
    # ------------------------------------------------------------------

    # Note: the code is in the .ipynb and in the README file
    # Try running the "ipython notebook" command to open the .ipynb notebook
