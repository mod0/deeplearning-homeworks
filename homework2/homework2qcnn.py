import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def cnn_model_mnist(features, labels, mode):
    """ Estimator for MNIST """

    # Input
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    # First convolution layer
    conv_layer_1 = tf.layers.conv2d(inputs=input_layer,
                                    filters=1,
                                    kernel_size=3,
                                    strides=1,
                                    padding="valid",
                                    activation=tf.nn.relu)
    # Output is 26x26
    # print(conv_layer_1.get_shape())
    
    # First pooling layer
    max_pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_1,
                                               pool_size=2,
                                               strides=2)
    # Output is 13x13
    # print(max_pool_layer_1.get_shape())
    
    # Second convolution layer
    conv_layer_2 = tf.layers.conv2d(inputs=max_pool_layer_1,
                                    filters=1,
                                    kernel_size=5,
                                    strides=1,
                                    padding="valid",
                                    activation=tf.nn.relu)
    # Output is 9x9
    # print(conv_layer_2.get_shape())
    
    # Second pooling layer
    max_pool_layer_2 = tf.layers.max_pooling2d(inputs=conv_layer_2,
                                               pool_size=2,
                                               strides=2)

    # Output is 4x4
    # print(max_pool_layer_2.get_shape())
    
    # Flattened layer
    flattened_layer_1 = tf.reshape(max_pool_layer_2, [-1, 4 * 4])

    # First dense layer
    dense_layer_1 = tf.layers.dense(inputs=flattened_layer_1, units=1024, activation=tf.nn.relu)

    
    # Final Softmax Layer - note no activation function is used here, its setup in the probabilities
    logits = tf.layers.dense(inputs=dense_layer_1, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # Convert the floating point labels to int32 and get the one hot encoding
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_args):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_mnist, model_dir='/tmp/mnist_conv_net')

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    mnist_classifier.train(input_fn= train_input_fn,
                           steps = 20000)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
      tf.app.run()
