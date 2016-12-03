#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np
import tensorflow as tf
import json
datX=[]
daty=[]
with open('vector_flatten.json') as json_data:
    
    datX.append(json.load(json_data))

with open('vector_flatten_label_nos.json') as json_data:
    
    daty.append(json.load(json_data))

print np.array(datX).shape               
X=np.array(datX).reshape((196,1961*170*1))
y=np.array(daty).reshape((196,20))
#X=np.transpose(X).reshape((1,950*170))               
#X=X[:,0:100]
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 5, 5, 1], padding='SAME')# try changing no. of strides
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # try changin the pooling options (strides and ksize)

nframes=1961 # No. of frames
#njoints=20 #No. of joints
jointdim=170 #Joint dimensions
nactivities=20 #No. of activities
conv1k=5 #convolution kernal size

x = tf.placeholder(tf.float32, shape=[None, jointdim*nframes*1])
y_ = tf.placeholder(tf.float32, shape=[None, nactivities])
#X=np.zeros((10,90*20*1))
#y=np.zeros((72,10)) # shd be removed once labes are done


# Time domain convolution
W_conv1 = weight_variable([conv1k, conv1k, 1, 256])# change no. of kernals for first conv
b_conv1 = bias_variable([256]) #change here too
x_image = tf.reshape(x, [-1,jointdim,nframes,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
(_,b,c,d)= h_pool1.get_shape()
print b.value


# First Fully connected layer
W_fc1 = weight_variable([b.value*c.value*d.value, 1024]) #change no. of neurons in fcl
b_fc1 = bias_variable([1024]) #change here too

h_pool1_flat = tf.reshape(h_pool1, [-1, b.value*c.value*d.value])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob1 = tf.placeholder(tf.float32) 
h_fc1_drop1 = tf.nn.dropout(h_fc1, keep_prob1) #shd check wat dropout does
print h_fc1_drop1.get_shape()


#second fully connected layer
W_fc2 = weight_variable([1024, 500])#Change this second values too
b_fc2 = bias_variable([500])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop1, W_fc2) + b_fc2)
keep_prob2 = tf.placeholder(tf.float32)
h_fc1_drop2 = tf.nn.dropout(h_fc2, keep_prob2)
print h_fc1_drop2.get_shape()


#Final output
W_fc3 = weight_variable([500, nactivities])#change here too
b_fc3 = bias_variable([nactivities])

y_conv = tf.matmul(h_fc1_drop2, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
bs=50 # change this to change the batch size
print "hi"                  
for i in range(100): # change the no. of iterations for training
    Xtr= X[bs*i:bs*(i+1),:]
    ytr= y[bs*i:bs*(i+1),:]
    #batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess,feed_dict={x:Xtr, y_: ytr, keep_prob1: 1.0, keep_prob2: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess,feed_dict={x: Xtr, y_: ytr, keep_prob1: 0.5, keep_prob2: 0.5})
print "hi"
#p= accuracy.eval(session=sess,feed_dict={ x: Xt, y_: yt, keep_prob1: 1.0, keep_prob2: 1.0})
