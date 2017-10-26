import numpy as np
import homework2q1 as hw2
params = hw2.initialize(eta = 0.01, maxiter = 500)
model  = hw2.sgd(**params)
# Total loss function value in an epoch is the sum of 
# loss functions corresponding to every example
print "Testing accuracy for least test loss: ", model['testing_accuracy_with_best_test_loss']
print "Weights corresponding to best testing accuracy: "
print "W1: ", model['w'][0]
print "W2: ", model['w'][1]
print "W3: ", model['w'][2]
# Note: In the following plots, we show how 
# training and testing loss changes with EPOCH.
# SGD algorithm is terminated once testing loss begins 
# to increase again
# hw2.doplots("Loss vs Epochs (training)", model['total_training_loss_fn_value'], "Training Loss", "Epochs", "Loss", semilogy = True)
# hw2.doplots("Loss vs Epochs (testing)", model['total_testing_loss_fn_value'], "Testing Loss", "Epochs", "Loss", semilogy = True)

# Find the f_score and other metrics
precision_recall_fscore = hw2.get_precision_recall_fscore(params['shuffle_order_testing'], params['test_data'], model['w'], params['z'], params['a'], params['y'])
print "Precision: ", precision_recall_fscore[0]
print "Recall: ", precision_recall_fscore[1]
print "F-Score: ", precision_recall_fscore[2]

# Validate using Keras model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

x_train     = params['train_data'][:,:-1]
y_train     = np.array([hw2.label_to_ytrue(2, x) for x in params['train_data'][:, -1]])
x_test      = params['test_data'][:,:-1]
y_test      = np.array([hw2.label_to_ytrue(2, x) for x in params['test_data'][:, -1]])

keras_model = Sequential()
keras_model.add(Dense(3, activation='relu', use_bias = False, kernel_initializer='random_uniform', input_dim = 2))
keras_model.add(Dense(3, activation='relu', use_bias = False, kernel_initializer='random_uniform'))
keras_model.add(Dense(2, activation='softmax', use_bias = False, kernel_initializer='random_uniform'))
sgd = SGD(lr=params['eta'])
keras_model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
keras_model.fit(x_train, y_train, batch_size = 1, epochs=params['maxiter'])
score = keras_model.evaluate(x_test, y_test, batch_size=1)
print score
kw = keras_model.get_weights()

print "Keras W1:", kw[0]
print "Keras W2:", kw[1]
print "Keras W3:", kw[2]

# Keras multiplies by weight vectors from the right so transposing it.
kwt = []
kwt.append(kw[0].T)
kwt.append(kw[1].T)
kwt.append(kw[2].T)

# Find the f_score and other metrics
precision_recall_fscore = hw2.get_precision_recall_fscore(params['shuffle_order_testing'], params['test_data'], kwt, params['z'], params['a'], params['y'])
print "Precision: ", precision_recall_fscore[0]
print "Recall: ", precision_recall_fscore[1]
print "F-Score: ", precision_recall_fscore[2]
