import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
What we'll do is
take the input data
send input to > hidden layer 1 in the way we weight it
From hidden layer 1 we run it through some actvation function
and then from there we'll send to hidden layer 2.
Now from hidden layer 2's activation function with weights we'll send it to output layer

When we pass data straight forward to very end is called as feed forward NN
and at end we compare Output with intended output and find the cost and loss function
eg of loss function can be cross entrophy

Then we'll use a  optimizer to minimze the cost (we'll use adam optimier, other are SDG etc)
What it does is it goes back and manipulate the weights - this is called as back propogation

feed forward + back propogation = epoch (1 cycle)
'''

mnist = input_data.read_data_sets('/tmp/data/', one_hot = True) # one hot means 1 is on others would be off

n_nodes_hl1 = 500 #hidden layer1 with 500 nodes
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 #it is going to go through batch of 100 features at a time and perform on it

#height * width
x = tf.placeholder('float', [None, 784]) #input data 784 pixel wide convert to 28*28
y = tf.placeholder('float') #we donot need the above value TF does on its own

def neural_network_model(data):
	#biases are added after the weights
	#input data * weights + biases
	#purpose of biases is used during activation function, so if all the input data is all 0 then no neuron will ever fire
	#so biase will add some value to fire a neuron in certain scenario
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),#initally starting with weight as 784(pixel value)
	'biases' : tf.Variable(tf.random_normal(n_nodes_hl1))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	'biases' : tf.Variable(tf.random_normal(n_nodes_hl2))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	'biases' : tf.Variable(tf.random_normal(n_nodes_hl3))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
	'biases' : tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #rectified linear activation function


	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2) #rectified linear activation function

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3) #rectified linear activation function

	output = tf.matmul(l3, output_layer['weights']) + hidden_3_layer['biases']

	return output
