from __future__ import division
import os
import io
import sys
import datetime
import numpy as np
import numpy.random as nr
import tensorflow as tf

sentiment = dict()
sentiment["postive"]  = np.array([1, 0]) # Is mispelled in the data file.
sentiment["negative"] = np.array([0, 1])

def read_word_vectors(filename):
    if os.path.isfile(filename):
        word_to_index = dict()
        index_to_word = list()
        word_vectors  = list()
        counter       = 0
        with open(filename,'r') as infile:
            for line in infile:
                parts = [x.strip() for x in line.split(",", 1)]
                word_to_index[parts[0]] = counter
                index_to_word.append(parts[0])
                word_vectors.append(np.genfromtxt(io.StringIO(unicode(parts[1], "utf-8")), delimiter=","))
                counter += 1
        return word_to_index, index_to_word, word_vectors
    else:
        raise Exception("Word vector file does not exist")

def read_data(filename):
    if os.path.isfile(filename):
        labels    = []
        data      = []
        maxsenlen = 0
        minsenlen = sys.maxsize
        with open(filename, 'r') as infile:
            for line in infile:
                parts = [x.strip() for x in line.split(",", 1)]
                labels.append(sentiment[parts[0]])
                data.append(parts[1])
                maxsenlen = np.max([maxsenlen, len(parts[1].split())])
                minsenlen = np.min([minsenlen, len(parts[1].split())])
        return data, labels, maxsenlen, minsenlen
    else:
        pass

def get_sentence_in_word_indices(sentence, word_to_index, max_sentence_length = 100):
    sentence_in_word_indices = np.zeros(max_sentence_length, dtype=np.int32)
    words_in_sentence = sentence.split()
    for i, word in enumerate(words_in_sentence):
        # Left pad with zero vectors
        sentence_in_word_indices[i + (max_sentence_length - len(words_in_sentence))] = word_to_index[word]
    return  sentence_in_word_indices

def main(wordvecfile = "sentiment-data/word-vectors-refine.txt",
         trainfile   = "sentiment-data/train.csv",
         testfile    = "sentiment-data/test.csv"):
    word_to_index, index_to_word, word_vectors = read_word_vectors(wordvecfile)
    word_vectors = np.array(word_vectors, dtype=np.float32)
    
    train_data, train_labels, maxsenlen, minsenlen = read_data(trainfile)
    train_labels                                   = np.array(train_labels)
    no_train_sentences                             = len(train_data)
    train_data_ints                                = np.zeros((no_train_sentences, maxsenlen), dtype=np.int32)
    print("Maximum sentence length in training data: ", maxsenlen)
    print("Minimum sentence length in training data: ", minsenlen)
    print("Total no. of sentences in training data : ", no_train_sentences)

    # convert each sentence into integer sequence
    for i, sentence in enumerate(train_data):
        train_data_ints[i, :] = get_sentence_in_word_indices(train_data[i], word_to_index, maxsenlen)
        
    test_data, test_labels, maxsenlen_test, minsenlen_test = read_data(testfile)
    test_labels                                            = np.array(test_labels)
    no_test_sentences                                      = len(test_data)
    test_data_ints                                         = np.zeros((no_test_sentences, maxsenlen), dtype=np.int32)

    assert(maxsenlen_test <= maxsenlen)

    print("Maximum sentence length in testing data: ", maxsenlen_test)
    print("Minimum sentence length in testing data: ", minsenlen_test)
    print("Total no. of sentences in testing data : ", no_test_sentences)
    
    # convert each test sentence into integer sequence
    for i, sentence in enumerate(test_data):
        test_data_ints[i, :] = get_sentence_in_word_indices(test_data[i], word_to_index, maxsenlen)
        
    # RNN Parameters
    batch_size     = 100
    n_tr_batches   = np.int(np.ceil(no_train_sentences/batch_size))

    # Split the training data into different batches
    train_data_indices   = np.arange(no_train_sentences)
    nr.shuffle(train_data_indices)
    train_data_indices   = np.array_split(train_data_indices, n_tr_batches)
    batched_train_data   = [train_data_ints[indices] for indices in train_data_indices]
    batched_train_labels = [train_labels[indices]    for indices in train_data_indices] 
    
    n_lstm_cell = 64
    n_classes   = 2
    maxiter     = 10
    wordvecdim  = 50

    # reset the default graph
    tf.reset_default_graph()

    # Create placeholder for labels
    t_labels  = tf.placeholder(tf.float32, [None, n_classes]) # labels
    t_data    = tf.placeholder(tf.int32,   [None, maxsenlen]) # training or test data

    
    # Create variables to hold the 3D tensor data of examples, words in sentences, word vectors
    indata = tf.nn.embedding_lookup(word_vectors, t_data)
    
    # Setup LSTM
    lstm_cell      = tf.nn.rnn_cell.LSTMCell(n_lstm_cell)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, indata, dtype=tf.float32)

    # weights for last softmax
    W = tf.Variable(tf.random_uniform([n_lstm_cell, n_classes]))
    b = tf.Variable(tf.constant(0.1, shape=[n_classes]))

    H          = tf.transpose(outputs, [1, 0, 2])
    h_final    = tf.gather(H, int(H.get_shape()[0]) - 1)
    prediction = tf.matmul(h_final, W) + b

    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(t_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=t_labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess  = tf.Session()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    for epoch in xrange(maxiter):
        for i in xrange(n_tr_batches):
            sess.run(optimizer, {t_data: batched_train_data[i],
                                 t_labels: batched_train_labels[i]})
        
        if ((epoch + 1) % 2 == 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=epoch)
            print("Saved checkpoint to %s" % save_path)
            
    print("Accuracy: ", sess.run(accuracy, feed_dict={t_data: test_data_ints,
                                                      t_labels: test_labels}))
    
if __name__ == "__main__":
    main()
