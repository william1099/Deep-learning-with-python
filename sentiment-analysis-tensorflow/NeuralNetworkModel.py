import tensorflow as tf;
import numpy as np;
from art3 import create_set;

train_x, train_y, test_x, test_y = create_set("pos.txt", "neg.txt");

n_nodes_h1 = 500;
n_nodes_h2 = 500;
n_nodes_h3 = 500;

n_class = 2;
batch_size = 100;

x = tf.placeholder(tf.float32, [None, len(train_x[0])]);
y = tf.placeholder(tf.float32);

def neural_network_model(data) :
    hidden_layer_1 = {'weights' : tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_h1])) ,
                      'bias' : tf.Variable(tf.random_normal([n_nodes_h1])) };

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_h2]))};

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_h3]))};

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_h3, n_class])),
                      'bias': tf.Variable(tf.random_normal([n_class]))};

    # (iinput_data * weight) + bias
    l1 = tf.add(tf.matmul(data, hidden_layer_1["weights"]) ,hidden_layer_1["bias"]);
    l1 = tf.nn.relu(l1);

    l2 = tf.add(tf.matmul(l1, hidden_layer_2["weights"]) , hidden_layer_2["bias"]);
    l2 = tf.nn.relu(l2);

    l3 = tf.add(tf.matmul(l2, hidden_layer_3["weights"]), hidden_layer_3["bias"]);
    l3 = tf.nn.relu(l3);

    output = tf.add(tf.matmul(l3, output_layer["weights"]), output_layer["bias"]);

    return output;

def train_network(x) :
    # training
    prediction = neural_network_model(x);
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y));
    optimizer = tf.train.AdamOptimizer().minimize(cost);
    n_epoch = 10;
    with tf.Session() as sse:
        sse.run(tf.global_variables_initializer());

        for epoch in range(n_epoch) :
            epoch_lost = 0;
            i = 0;
            while i < len(train_x) :
                start = i;
                end = i + batch_size;
                i += batch_size;

                train_xx = train_x[start:end];
                train_yy = train_y[start:end];

                _, c = sse.run([optimizer, cost], feed_dict={x: train_xx, y: train_yy});
                epoch_lost += c;
                print("epoch ", epoch, " completed out of ", n_epoch, "epoch_lost ", epoch_lost);

        # testing
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1));  # return [1,0,1,0]
        accuracy = tf.reduce_mean(tf.cast(correct, "float"));
        print("accuracy : ", accuracy.eval({x:test_x, y:test_y}));

train_network(x); # accuracy  = 56 %
