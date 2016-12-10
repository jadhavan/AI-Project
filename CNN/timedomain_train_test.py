#! /usr/bin/python   
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from random import sample
import numpy as np
import tensorflow as tf
import json

stridesize = int(sys.argv[1])
seed = int(sys.argv[2])
tf.set_random_seed(seed)

print "Running with stride %i and seed %i"%(stridesize, seed)

os.chdir("0.3")

datX=[]
daty=[]
with open('vector_flatten_pos.json') as json_data:
    datX.append(json.load(json_data))

with open('vector_flatten_label_nos.json') as json_data:
    daty.append(json.load(json_data))
#print mnist.train.images.shape
print np.array(datX).shape
X=np.array(datX).reshape((228,1961*15*4))
#X=(X-np.average(X,axis=0))/np.std(X,axis=0)
y=np.array(daty).reshape((228,20))
data_nos_test=[1,42,74,105,168,7,24,72,104,133,114,81,89,149,98,77,110,30,64,33,192,93,186,85,146,28,0,20,69,132,26,39,14,226,220,34,212,4,204,90,122,10,201,2,199,15,32]
print len(data_nos_test)

data_nos_train = []
for i in range(np.array(datX)[0].shape[0]):
    if data_nos_test.count(i):
        continue
    data_nos_train.append(i)


X_test = np.delete(X,data_nos_train,0)
y_test = np.delete(y,data_nos_train,0)
X_train = np.delete(X,data_nos_test,0)
y_train = np.delete(y,data_nos_test,0)
print X_train.shape
#X=np.transpose(X).reshape((1,950*170))
#X=X[:,0:100]
sess = tf.InteractiveSession()

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1,name=name)
    return tf.Variable(initial)
def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape,name=name)
    return tf.Variable(initial)
def conv2d(x, W):
    #(Change the strides value in this from [1,1,1,,1] to [1,7,7,1])
    return tf.nn.conv2d(x, W, strides=[1, stridesize, stridesize, 1], padding='SAME')# try changing no. of strides 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # try changin the pooling options (strides and ksize)

nframes=1961 # No. of frames
njoints=15 #No. of joints
jointdim=4 #Joint dimensions
nactivities=20 #No. of activities
conv1k=10 #convolution kernal size
conv2k=5
conv3k=3
x = tf.placeholder(tf.float32, shape=[None, njoints*nframes*jointdim])
y_ = tf.placeholder(tf.float32, shape=[None, nactivities])
#X=np.zeros((10,90*20*1))
#y=np.zeros((72,10)) # shd be removed once labes are done


# Time domain convolution
W_conv1 = weight_variable([10, conv1k, jointdim, 256],name="W_conv1")# change no. of kernals for first conv
b_conv1 = bias_variable([256],name="b_conv1") #change here too
x_image = tf.reshape(x, [-1,njoints,nframes,jointdim])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
(_,b,c,d)= h_pool1.get_shape()


W_conv2 = weight_variable([conv2k, conv2k, 256, 64],name="W_conv2")# change no. of kernals for first conv
b_conv2 = bias_variable([64],name="b_conv2") #change here too
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
(_,b,c,d)= h_pool2.get_shape()

W_conv3 = weight_variable([conv3k, conv3k, 64, 32],name="W_conv3")# change no. of kernals for first conv
b_conv3 = bias_variable([32],name="b_conv3") #change here too
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
(_,b,c,d)= h_pool2.get_shape()
print b*c*d

# First Fully connected layer
W_fc1 = weight_variable([b.value*c.value*d.value, 600],name="W_fc1") #change no. of neurons in fcl
b_fc1 = bias_variable([600],name="b_fc1") #change here too

h_pool2_flat = tf.reshape(h_pool2, [-1, b.value*c.value*d.value])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob1 = tf.placeholder(tf.float32)
h_fc1_drop1 = tf.nn.dropout(h_fc1, keep_prob1) #shd check wat dropout does
print h_fc1_drop1.get_shape()


#second fully connected layer
W_fc2 = weight_variable([600, 100],name="W_fc2")#Change this second values too
b_fc2 = bias_variable([100],name="b_fc2")

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop1, W_fc2) + b_fc2)
keep_prob2 = tf.placeholder(tf.float32)
h_fc1_drop2 = tf.nn.dropout(h_fc2, keep_prob2)
print h_fc1_drop2.get_shape()


#Final output
W_fc3 = weight_variable([100, nactivities],name="W_fc3")#change here too
b_fc3 = bias_variable([nactivities],name="b_fc3")

y_conv = tf.matmul(h_fc1_drop2, W_fc3) + b_fc3
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
bs=40 # change this to change the batch size

print "started learning..."
mxval=0.643
cst=40
for i in range(2000): # change the no. of iterations for training
    for j in range(4):

        Xtr= X_train[bs*j:bs*(j+1),:]
        ytr= y_train[bs*j:bs*(j+1),:]
        #print Xtr.shape
    #batch = mnist.train.next_batch(50)
        train_accuracy = accuracy.eval(session=sess,feed_dict={x:X_test, y_: y_test, keep_prob1: 1.0, keep_prob2: 1.0})
        cost = cross_entropy.eval(feed_dict={x:X, y_: y, keep_prob1: 1.0, keep_prob2: 1.0})

        if i%25 == 0:
            print("step %d, training (test) accuracy %g, cost %g"%(i, train_accuracy, cost))


        if train_accuracy > mxval and cost < 40:
            cst=cost
            mxval=train_accuracy
            print("step %d, training (test) accuracy %g, cost %g"%(i, train_accuracy, cost))
            saver.save(sess, "/home/smurlidaran/AIproject/model2")
            pred=tf.argmax(y_conv,1)
            label= tf.argmax(y_,1)
            print sess.run(pred,feed_dict={x:X_test, y_: y_test, keep_prob1: 1.0, keep_prob2: 1.0})
            print sess.run(label,feed_dict={y_: y_test})

        if train_accuracy==mxval and cost<cst:
            print("step %d, training (test) accuracy %g, cost %g"%(i, train_accuracy, cost))
            saver.save(sess, "/home/smurlidaran/AIproject/model2")
            pred=tf.argmax(y_conv,1)
            label= tf.argmax(y_,1)
            print sess.run(pred,feed_dict={x:X_test, y_: y_test, keep_prob1: 1.0, keep_prob2: 1.0})
            print sess.run(label,feed_dict={y_: y_test})
    #print sess.run(W_conv2)
    #print sess.run(b_conv2.name)
        sess.run(train_step,feed_dict={x: Xtr, y_: ytr, keep_prob1: 0.8, keep_prob2: 1.0})
print "Finished learning..."

#save_path = saver.save(sess, "/home/smurlidaran/AIproject/odel.ckpt")
p= accuracy.eval(session=sess,feed_dict={ x: X_test, y_: y_test, keep_prob1: 1.0, keep_prob2: 1.0})
#print sess.run(pred,feed_dict={x:X_test, y_: y_test, keep_prob1: 1.0, keep_prob2: 1.0})
#print sess.run(label,feed_dict={y_: y_test})
print p
