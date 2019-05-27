import tensorflow as tf
import numpy as np
import os
import codecs
import random


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
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
    n_values = 117 + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


batch_size = 512
X_ = tf.placeholder(tf.float32, [None, 6, 128, 1],name='cnn_X')
label_ = tf.placeholder(tf.float32, [None, 118],name='cnn_Y')

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
'''
'''
	pooling
	6*32*128->6*16*128
'''
h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1],padding='VALID')
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
cnn_output = tf.multiply(h_conv4,1,name='cnn_output')
W_fc = weight_variable([2048, 118])
b_fc = bias_variable([118])
h_fc = tf.nn.softmax(tf.matmul(h_flat, W_fc) + b_fc)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_ * tf.log(h_fc+1e-10), reduction_indices=[1]),name='cnn_loss')
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_fc, 1), tf.argmax(label_, 1),name='cnn_pre_Y')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='cnn_accuracy')

X_train = load_X('./123')
X_test = load_X('./123')

train_label = load_y('./y_train.txt')
test_label = load_y('./y_test.txt')

saver = tf.train.Saver(max_to_keep=1)

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer()) 

best_accuracy = 0
wtd=0
if wtd==1:
    for i in range(200):
        l = len(train_label)
        batch_idxs = l / batch_size
        index = range(l)
        random.shuffle(index)
        for idx in range(batch_idxs):
            image_idx = X_train[index[idx * batch_size:(idx + 1) * batch_size]]
            label_idx = train_label[index[idx * batch_size:(idx + 1) * batch_size]]
            #print(start,end)
            acc, loss, _ = sess.run([accuracy, cross_entropy, train_step], feed_dict={
                    X_: image_idx,
                    label_: label_idx
                    })
            if idx % 100 == 0:
                print(str(i) + 'the cross_entropy:', str(loss), 'train_accuracy:', str(acc))
                # Test completely at every epoch: calculate accuracy
                accuracy_out, loss_out = sess.run(
                        [accuracy, cross_entropy],
                        feed_dict={
                                X_: X_test,
                                label_: test_label
                                }
                        )
                if accuracy_out > best_accuracy:
                    saver.save(sess,'./cnn_ckpt/model')
                    best_accuracy = accuracy_out
                    print(str(i)+'--------------the cross_entropy:', str(loss_out), '-----------------------test_accuracy:', str(accuracy_out))
                    
    print("best accuracy:"+str(best_accuracy))
else:
    model_file=tf.train.latest_checkpoint('cnn_ckpt/')
    print(model_file)
    saver.restore(sess,model_file)
    t=len(train_label)
    f=[]
    for j in range(256):
        f.append(open("train_"+str(j)+".txt",'w'))
    for i in range(t):
        batch_xs=[]
        li=X_test[i][:,:128]
        batch_xs.append(li)
        batch_ys=[]
        batch_ys.append(train_label[i])
        val_loss_1=sess.run([h_conv4], feed_dict={X_:batch_xs, label_:batch_ys})
        batch_xs=[]
        li=X_test[i][:,128:]
        batch_xs.append(li)
        batch_ys=[]
        batch_ys.append(train_label[i])
        val_loss_2=sess.run([h_conv4], feed_dict={X_:batch_xs, label_:batch_ys})
        print("open"+str(i))
        for j in range(128):   
            for k in range(15):
                f[j].write(str(val_loss_1[0][0][0][k][j])+" ")
            f[j].write(str(val_loss_1[0][0][0][15][j]))
            f[j].write("\n")
            for l in range(15):
                f[j+128].write(str(val_loss_2[0][0][0][l][j])+" ")
            f[j+128].write(str(val_loss_2[0][0][0][15][j]))
            f[j+128].write("\n")
    for j in range(256):
        f[j].close()
'''
        print(len(val_loss_1))
        print(len(val_loss_1[0]))
        print(len(val_loss_1[0][0][0]))
        print(len(val_loss_1[0][0][0][0]))
        所以格式是1*1*1*16*128
'''

'''
        for j in val[0].tolist():
            f.write(str(j)+" ")
'''


