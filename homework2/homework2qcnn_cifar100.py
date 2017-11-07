import os
import cPickle
import matplotlib
import numpy as np
import tensorflow as tf
import numpy.random as nr
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_cifar100(features, labels, mode):
    """ Estimator for CIFAR 100 """

    # Input: is already reshaped while reading
    input_layer = features['x']
    
    # First convolution layer
    conv_layer_1 = tf.layers.conv2d(inputs=input_layer,
                                    filters=64,
                                    kernel_size=3,
                                    strides=1,
                                    padding="valid",
                                    activation=tf.nn.relu)
    # Output is 30x30
    # print(conv_layer_1.get_shape())
    
    # First pooling layer
    max_pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_1,
                                               pool_size=2,
                                               strides=2)
    # Output is 15x15
    # print(max_pool_layer_1.get_shape())
    
    # Second convolution layer
    conv_layer_2 = tf.layers.conv2d(inputs=max_pool_layer_1,
                                    filters=64,
                                    kernel_size=5,
                                    strides=1,
                                    padding="valid",
                                    activation=tf.nn.relu)
    # Output is 11x11
    # print(conv_layer_2.get_shape())
    
    # Second pooling layer
    max_pool_layer_2 = tf.layers.max_pooling2d(inputs=conv_layer_2,
                                               pool_size=2,
                                               strides=2)

    # Output is 5x5
    # print(max_pool_layer_2.get_shape())
    
    # Flattened layer
    flattened_layer_1 = tf.reshape(max_pool_layer_2, [-1, 5 * 5 * 64])

    # First dense layer
    dense_layer_1 = tf.layers.dense(inputs=flattened_layer_1, units=1024, activation=tf.nn.relu)

    
    # Final Softmax Layer - note no activation function is used here, its setup in the
    # loss function where softmax cross entropy is used and in the probabilities under
    # predictions dictionary in PREDICT mode.
    logits = tf.layers.dense(inputs=dense_layer_1, units=100)

    predictions = {
        "class": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # Convert the floating point labels to int32 and get the one hot encoding
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=100)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        test_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["class"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=test_metric_ops)

def load_cifar100_data(filename):
    # Check if file exists
    if os.path.isfile(filename):
        with open(filename, 'rb') as infile:
            unpickledata = cPickle.load(infile)
        image_data    = unpickledata['data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
        image_labels  = unpickledata['fine_labels']

        return image_data, image_labels
    else:
        raise Exception("CIFAR-100 data file does not exist")
        
def load_cifar100_label_names(filename):
    # Check if file exists
    if os.path.isfile(filename):
        with open(filename, 'rb') as infile:
            unpickledata = cPickle.load(infile)

        label_names = unpickledata['fine_label_names']

        return label_names
    else:
        raise Exception("CIFAR-100 meta file does not exist")
    

def main(*args, **kwargs):   
    # Load training and eval data
    # Give full path
    if not "trainfile" in kwargs:
        trainfile = 'cifar100-data/train'
    else:
        trainfile = kwargs["trainfile"]
        
    if not "testfile"  in kwargs:
        testfile  = 'cifar100-data/test'
    else:
        testfile  = kwargs["testfile"]

    if not "metafile" in kwargs:
        metafile = 'cifar100-data/meta'
    else:
        metafile = kwargs["metafile"]
        
    train_data, train_labels = load_cifar100_data(trainfile)
    train_data = np.asarray(train_data, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int32)
    test_data, test_labels = load_cifar100_data(testfile)
    test_data = np.asarray(test_data, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.int32)

    label_names = load_cifar100_label_names(metafile)

    no_test_images = len(test_labels)

    # Create the Estimator
    cifar100_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_cifar100, model_dir='/tmp/cifar100_conv_net')

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    cifar100_classifier.train(input_fn= train_input_fn,
                           steps = 20000)
    
    # Evaluate the model and print results
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    test_results = cifar100_classifier.evaluate(input_fn=test_input_fn)
    print(test_results)

    predict_indices = nr.randint(0, no_test_images, size=(9,))
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data[predict_indices]},
        num_epochs=1,
        shuffle=False)
    prediction_results = list(cifar100_classifier.predict(input_fn=predict_input_fn))

    predicted_images = test_data[predict_indices].reshape(-1, 32, 32, 3).squeeze()
    
    fig, ax = plt.subplots(3, 3)
    for i in xrange(3):
        for j in xrange(3):
             ax[i][j].imshow(predicted_images[i*3 + j])
             ax[i][j].set_title("True: {}  Predicted: {}".format(label_names[test_labels[predict_indices[i*3 + j]]], label_names[prediction_results[i*3 + j]['class']]))
    fig.show() 


if __name__ == "__main__":
      tf.app.run()
