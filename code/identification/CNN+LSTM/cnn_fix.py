import tensorflow as tf
import numpy as np
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def load_X(path):
    X_signals = []
    files = os.listdir(path)
    files.sort(key=str.lower)
    #file
    #['train_acc_x.txt', 'train_acc_y.txt', 'train_acc_z.txt', 'train_gyr_x.txt', 'train_gyr_y.txt', 'train_gyr_z.txt']
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
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#----------------------------------the part of LSTM--------------------------------
def LSTM_Network(input):
    n_hidden = 1024
    n_steps = 128
    n_inputs = 6
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(input, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_inputs])
    # new shape: (n_steps*batch_size, n_input)
    W = tf.Variable(tf.random_normal([n_inputs, n_hidden]))
    B = tf.Variable(tf.random_normal([n_hidden], mean=1.0)),
    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, W) + B)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X,n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page

    # Linear activation
    #return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']
    return  outputs[-1]

def last_full_connection_layer(lstm_output,cnn_output):
    cnn_output = tf.contrib.layers.flatten(cnn_output)
    eigen_input = tf.concat([lstm_output, cnn_output],1)
    W_fc2 = weight_variable([1024+2048, 118])
    b_fc2 = bias_variable([118])
    #y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    return tf.nn.softmax(tf.matmul(eigen_input, W_fc2) + b_fc2)

X_ = tf.placeholder(tf.float32, [None, 128, 6])
label_ = tf.placeholder(tf.float32, [None, 118])


X_train = load_X('../data/train/record')
X_test = load_X('../data/test/record')

train_label = load_y('../data/train/label.txt')
test_label = load_y('../data/test/label.txt')

sess = tf.InteractiveSession(config=config)
#lstm
lstm_output = LSTM_Network(X_)
#cnn
# load model
cnn_saver = tf.train.import_meta_graph('./cnn_ckpt/model.meta')
cnn_saver.restore(sess, tf.train.latest_checkpoint('./cnn_ckpt/'))
graph = tf.get_default_graph()
# cnn placeholder
cnn_X = graph.get_tensor_by_name("cnn_X:0")
# cnn output
cnn_output = graph.get_tensor_by_name("cnn_output:0")

pred_Y = last_full_connection_layer(lstm_output,cnn_output)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_ * tf.log(pred_Y+1e-10), reduction_indices=[1]),name='cnn_fix_cross_entropy')
train_step = tf.train.AdamOptimizer(1e-3,name='cnn_fix_train_step').minimize(cross_entropy,name='cnn_fix_minimize')
correct_prediction = tf.equal(tf.argmax(pred_Y,1), tf.argmax(label_,1),name='cnn_fix_correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32,name='cnn_fix_accuracy'))

sess.run(tf.global_variables_initializer())
f = open('result_cnn_fix.txt','w')
best_accuracy = 0
for i in range(100):
    batch_size = 128
    for start,end in zip(range(0,len(train_label),batch_size),
                         range(batch_size,len(train_label)+1,batch_size)):
        cnn_feed = np.transpose(X_train[start:end],[0,2,1]).reshape([-1,6,128,1])
        sess.run(train_step,feed_dict={
            cnn_X:cnn_feed,
            X_:X_train[start:end],
            label_:train_label[start:end]
        })
        # Test completely at every epoch: calculate accuracy
    cnn_test_feed = np.transpose(X_test, [0, 2, 1]).reshape([-1, 6, 128, 1])
    accuracy_out, loss_out = sess.run(
        [accuracy, cross_entropy],
        feed_dict={
            cnn_X: cnn_test_feed,
            X_:X_test,
            label_:test_label
        }
    )
    if accuracy_out > best_accuracy:
        best_accuracy = accuracy_out
    print(str(i)+'th cross_entropy:',str(loss_out),'accuracy:',str(accuracy_out))
    f.write(str(i)+'th cross_entropy:',str(loss_out),'accuracy:',str(accuracy_out))
print("best accuracy:"+str(best_accuracy))
f.write("best accuracy:"+str(best_accuracy))
f.close()
