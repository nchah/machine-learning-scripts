# nchah/machine-learning-scripts/tensorflow

## tensorflow-mnist

Running the tutorial scripts on the command line.

```
$ source activate tensorflow

# The sample code as is
(tensorflow) $ python mnist_softmax.py

Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
0.918


# The same code but written while going through the guide
(tensorflow) $ python mnist_softmax_beginner.py

Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
MNIST data loaded.
Starting to build the TensorFlow computation graph...
Starting training stage...
Training the model 1000 times...
Evaluating the model...
Accuracy: 0.9154


# The Expert tutorial where a multilayer convolutional network is added
(tensorflow) $ python mnist_softmax_expert.py

Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
MNIST data loaded.
Starting to build the TensorFlow computation graph...
Starting training stage...
Training the model 1000 times...
Evaluating the model...
Accuracy: 0.9154
Implementing Multilayer Convolutional Network to improve accuracy.
Training the model...
step 0, training accuracy 0.14
step 100, training accuracy 0.82
step 200, training accuracy 0.9
step 300, training accuracy 0.92
step 400, training accuracy 0.98
step 500, training accuracy 0.94
step 600, training accuracy 0.92
step 700, training accuracy 0.96
step 800, training accuracy 0.94
step 900, training accuracy 0.96
step 1000, training accuracy 0.94
step 1100, training accuracy 0.94
step 1200, training accuracy 1
step 1300, training accuracy 0.98
...

```

## word2vec

Further work with word2vec done in this repository: [nchah/word2vec4everything](https://github.com/nchah/word2vec4everything)




