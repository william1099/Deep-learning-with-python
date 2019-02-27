# Sentiment Analysis with Tensorflow

Sentiment analysis is one of the most common NLP methodology aimed at extracting insight from social data. It is really powerful
to gain an understanding of people's opinion, emotion expressed behind certain topic. Sentiment analysis can be used to solve 
language-specific kind of problem. Some implementations are to help business identify people's sentiment of their brand, to 
determine the expression of a movie review which is what this code aimed at solving

## Documentation 

In this code, we will build a neural network model (multi-layer perceptron) with tensorflow and apply the sentiment data training and analysis to 
the model. This neural network model is composed of 3 hidden layers where each of them has 500 nodes. The output has 2 nodes which 
will identify if the sentiment is positive or negative.

```python

n_nodes_h1 = 500;
n_nodes_h2 = 500;
n_nodes_h3 = 500;
n_class = 2;
    
    hidden_layer_1 = {'weights' : tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_h1])) ,
                      'bias' : tf.Variable(tf.random_normal([n_nodes_h1])) };

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_h2]))};

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_h3]))};

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_h3, n_class])),
                      'bias': tf.Variable(tf.random_normal([n_class]))};
```

In preprocess.py, we will extract and preprocess the datas (pos.txt and neg.txt) and transform them into vectors which would then feed the 
network model

## Result

The accuracy achieved during testing is about 58% which is far from perfect. While it is not perfect, it can be used to help at
understanding how sentiment analysis works and how to build neural network model to apply them. if you want to get a better 
accuracy, you can use much more dataset or implement RNN or CNN model to it.
