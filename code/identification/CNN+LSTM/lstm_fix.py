import tensorflow as tf
import numpy as np
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def load_X(path):
    X_signals = []
    files = os.listdir(path)
    files.sort(key=str.lower)

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

def CNN_NetWork(X_):
    CNN_input = tf.transpose(X_,[0,2,1])
    CNN_input = tf.reshape(CNN_input, [-1, 6, 128, 1])
    # input shape [batch, in_height, in_width, in_channels]
    # kernel shape [filter_height, filter_width, in_channels, out_channels]
    '''
    	1*9
    	stride = 2 
    	padding
    	6*128->6*64*32
    '''
    W_conv1 = weight_variable([1, 9, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(CNN_input, W_conv1, strides=[1, 1, 2, 1], padding='SAME') + b_conv1)
    '''
    	pooling
    	6*64*32->6*32*32
    '''
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
    '''
    	1*3
    	stride = 1
    	6*32*32->6*32*64
    '''
    W_conv2 = weight_variable([1, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
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

    '''
    	pooling
    	6*32*128->6*16*128
    '''
    h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
    '''
    	6*1
    	6*32*128->1*16*128
    '''
    W_conv4 = weight_variable([6, 1, 128, 128])
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(
        tf.nn.conv2d(h_pool2, W_conv4, strides=[1, 1, 1, 1], padding='VALID') + b_conv4)
    '''
    	input flat 16*128=2048
    	output 118
    '''
    h_flat = tf.contrib.layers.flatten(h_conv4)

    return h_flat

def last_full_connection_layer(lstm_output,cnn_output):
    eigen_input = tf.concat([lstm_output, cnn_output],1)

    W_fc2 = weight_variable([1024+2048, 118])
    b_fc2 = bias_variable([118])
    #y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    return tf.nn.softmax(tf.matmul(eigen_input, W_fc2) + b_fc2)

class model:
    def __init__(self,ckpt_dir,meta_name,placeholder,output):
        self.ckpt_dir = ckpt_dir
        self.model_name = meta_name
        self.graph = tf.Graph()
        self.sess = tf.Session(config=config, graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver = tf.train.import_meta_graph(os.path.join(self.ckpt_dir,self.model_name))
                self.saver.restore(self.sess,tf.train.latest_checkpoint(self.ckpt_dir))
                self.X = self.sess.graph.get_tensor_by_name(placeholder)
                self.output = self.sess.graph.get_tensor_by_name(output)
    def produce(self,data):
        with self.sess.as_default():
            with self.graph.as_default():
                return self.sess.run(self.output,feed_dict={self.X:data})

X_ = tf.placeholder(tf.float32, [None, 128, 6])
label_ = tf.placeholder(tf.float32, [None, 118])

X_train = load_X('../data/train/record')
X_test = load_X('../data/test/record')

train_label = load_y('../data/train/label.txt')
test_label = load_y('../data/test/label.txt')

batch_size = 512

lstm_output = tf.placeholder(tf.float32,[None,1024],name='lstm_output')
#lstm
'''
sess = tf.InteractiveSession(config=config)
lstm_saver = tf.train.import_meta_graph('./lstm_1024_ckpt/lstm_model.meta')
lstm_saver.restore(sess, tf.train.latest_checkpoint('./lstm_1024_ckpt/'))

graph = tf.get_default_Graph()
# cnn placeholder
lstm_X = graph.get_tensor_by_name("lstm_X:0")
# cnn output
lstm_output = graph.get_tensor_by_name("lstm_output:0")
'''

lstm_model = model('./lstm_1024_ckpt/','lstm_model.meta','lstm_X:0','lstm_output:0')

#cnn
cnn_output = CNN_NetWork(X_)
pred_Y = last_full_connection_layer(lstm_output, cnn_output)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_ * tf.log(pred_Y+1e-10), reduction_indices=[1]),name='lstm_fix_cross_entropy')
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy, name='lstm_fix_train_global_step')
correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(label_,1),name='lstm_fix_correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='lstm_fix_accuracy')

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)
saver.restore(sess,tf.train.latest_checkpoint('./lstm_fix_ckpt/'))
best_accuracy = 0
f = open('result_lstm_fix.txt','w')
for i in range(100):
    l = len(train_label)
    batch_idxs = int(l / batch_size)
    index = list(range(l))
    random.shuffle(index)
    for idx in range(batch_idxs):
        image_idx = X_train[index[idx * batch_size:(idx + 1) * batch_size]]
        label_idx = train_label[index[idx * batch_size:(idx + 1) * batch_size]]
        # produce data for last full connection
        data4lstm = lstm_model.produce(image_idx)

        sess.run(train_step,feed_dict={
            lstm_output:data4lstm,
            X_:image_idx,
            label_:label_idx
        })
    # Test completely at every epoch: calculate accuracy

    test4lstm = lstm_model.produce(X_test)
    accuracy_out, loss_out = sess.run(
        [accuracy, cross_entropy],
        feed_dict={
            lstm_output: test4lstm,
            X_:X_test,
            label_:test_label
        }
    )
    if accuracy_out > best_accuracy:
        saver.save(sess,'./lstm_fix_ckpt/lstm_model')
        best_accuracy = accuracy_out
    print(str(i)+'th cross_entropy:',str(loss_out),'test_accuracy:',str(accuracy_out))
    f.write(str(i) + '  the cross_entropy: ' + str(loss_out) + '  test_accuracy:' + str(accuracy_out) + '  \n')

print("best accuracy:"+str(best_accuracy))
f.write("best accuracy:"+str(best_accuracy)+ '  \n')
f.close()
