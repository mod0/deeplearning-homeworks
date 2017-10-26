import os
import io
import numpy as np

sentiment = dict()
sentiment["postive"]  = 1  # Is mispelled in the data file.
sentiment["negative"] = 0

def read_word_vectors(filename):
    if os.path.isfile(filename):
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

