import tensorflow as tf
import numpy as np
import os

#load数据
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
    return np.transpose(np.array(X_signals), (1, 2, 0))#(totalStepNum*128*6)
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
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
#---------------------------the part of CNN---------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#----------------------------------the part of LSTM--------------------------------
class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.n_layers = 2   # nb of layers
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = 118  # Final output classes
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
    #return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']
    return  lstm_last_output

def CNN_NetWork(X_):
    CNN_input = tf.reshape(X_,[-1, 128, 6, 1])
    # 第一个卷积层&pooling,h_pool1(?*64*6*32通道)
    W_conv1 = weight_variable([3, 3, 1, 32])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目(卷积核个数)
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.elu(
        tf.nn.conv2d(CNN_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第二个卷积层&pooling,(?*32*6*64通道)
    W_conv2 = weight_variable([3, 3, 32, 64])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目(卷积核个数)
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.elu(
        tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME')

    # 第三层 是个全连接层,输入维数32*3*64, 输出维数为64
    W_fc1 = weight_variable([32 * 3 * 64, 64])
    b_fc1 = bias_variable([64])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 3 * 64])
    h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1

def last_full_connection_layer(lstm_output,cnn_output):
    eigen_input = tf.concat([lstm_output, cnn_output],1)
    # 第四层，输入64维，输出118维，也就是具体的分类
    W_fc2 = weight_variable([128, 118])
    b_fc2 = bias_variable([118])
    #y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)  # 使用softmax作为多分类激活函数
    return tf.nn.softmax(tf.matmul(eigen_input, W_fc2) + b_fc2)  # 使用softmax作为多分类激活函数


X_ = tf.placeholder(tf.float32, [None, 128, 6])  # 输入占位
label_ = tf.placeholder(tf.float32, [None, 118])  # label占位

#输入
X_train = load_X('./data/train/record')
X_test = load_X('./data/test/record')
#得到label
train_label = load_y('./data/train/label.txt')
test_label = load_y('./data/test/label.txt')

config = Config(X_train, X_test)
lstm_output = LSTM_Network(X_,config)
cnn_output = CNN_NetWork(X_)
pred_Y = last_full_connection_layer(lstm_output,cnn_output)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_ * tf.log(pred_Y+1e-10), reduction_indices=[1])) # 损失函数，交叉熵
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) # 使用adam优化
correct_prediction = tf.equal(tf.argmax(pred_Y,1), tf.argmax(label_,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) # 变量初始化

best_accuracy = 0
for i in range(100):
    batch_size = 512
    for start,end in zip(range(0,len(train_label),batch_size),
                         range(batch_size,len(train_label)+1,batch_size)):
        sess.run(train_step,feed_dict={
            X_:X_train[start:end],
            label_:train_label[start:end]
        })
        # Test completely at every epoch: calculate accuracy
    accuracy_out, loss_out = sess.run(
        [accuracy, cross_entropy],
        feed_dict={
            X_:X_test,
            label_:test_label
        }
    )
    if accuracy_out > best_accuracy:
        best_accuracy = accuracy_out
    print(str(i)+'th cross_entropy:',str(loss_out),'accuracy:',str(accuracy_out))

print("best accuracy:"+str(best_accuracy))