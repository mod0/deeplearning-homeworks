import os
import io
import numpy as np

sentiment = dict()
sentiment["postive"]  = 1  # Is mispelled in the data file.
sentiment["negative"] = 0

def read_word_vectors(filename):
    if os.path.isfile(filename):
        # TODO: Change this Raphael
        # See below I am creating a dictionary right?
        # In the dictionary I am storing:
        # the word as the key
        # and the array of word vector as value for easy retrieval
        # Instead of that you need to create the following two lists
        # 1) a mapping (dictionary) from word to the index in the array at which you saw it,
        # for this you need a counter that you increment for each new word that
        # you will store in a new dictionary with the word as the key
        # 2) a list of corresponding word vectors
        # Below in the code, parts[0] is the word
        # And np.genfromtxt(...) will give you the numpy array of word vector
        # corresponding to the word
        # You will use that information to construct the two data structures that
        # I mentioned and increment your counter for each word you see.
        # Finally, you will return both of these to the main routine.
        wordvecdict = dict()
        with open(filename,'r') as infile:
            for line in infile:
                parts = [x.strip() for x in line.split(",", 1)]
                wordvecdict[parts[0]] = np.genfromtxt(io.StringIO(unicode(parts[1], "utf-8")), delimiter=",")
        return wordvecdict
    else:
        raise Exception("Word vector file does not exist")

def read_data(filename):
    if os.path.isfile(filename):
        labels = []
        data   = []
        with open(filename, 'r') as infile:
            for line in infile:
                parts = [x.strip() for x in line.split(",", 1)]
                labels.append(sentiment[parts[0]])
                data.append(parts[1])
        return data, labels
    else:
        pass

def get_sentence_in_wordvectors(sentence, wordvecdict, wordvector_length = 50, max_sentence_length = 100, zerofill = True):
    # TODO: Change this Raphael
    # Instead of returning the sentence with word vectors
    # only return the list of words as integers using the mapping (dictionary)
    # you created in the previous read_word_vectors function.
    # Then tensorflow can lookup the corresponding word vectors automatically
    # using tf.nn.embedding lookup or something
    sentence_in_wordvectors = np.zeros((50, 100))
    for i, word in enumerate(sentence.split()):
        sentence_in_wordvectors[:, i] = wordvecdict[word]
    return  sentence_in_wordvectors


if __name__ == "__main__":
    wordvecdict = read_word_vectors("sentiment-data/word-vectors-refine.txt")
    print(wordvecdict["fawn"])
    print(len(wordvecdict.keys()))

    train_data, train_labels = read_data("sentiment-data/train.csv")
    print(train_data[0])
    print(train_labels[0])
    print(len(train_data))
    print(len(train_labels))

    test_data, test_labels = read_data("sentiment-data/test.csv")
    print(test_data[0])
    print(test_labels[0])
    print(len(test_data))
    print(len(test_labels))

    sentence = get_sentence_in_wordvectors(train_data[0], wordvecdict)
    print(len(train_data[0].split()))
    print(sentence.shape)
    print(sentence)

