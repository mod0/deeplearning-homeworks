import os
import cPickle
import numpy as np
import tensorflow as tf

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
        "classes": tf.argmax(input=logits, axis=1),
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
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

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
    

def main(unused_args):   
    # Load training and eval data
    # Give full path
    train_data, train_labels = load_cifar100_data('train')
    train_data = np.asarray(train_data, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_data, eval_labels = load_cifar100_data('test')
    eval_data = np.asarray(eval_data, dtype=np.float32)
    eval_labels = np.asarray(eval_labels, dtype=np.int32)

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
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = cifar100_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
      tf.app.run()
