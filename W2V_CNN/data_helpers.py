import numpy as np
import re
import itertools
from collections import Counter
import csv
import math
import pickle

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    f_data = open('data.txt', 'rb')
    x_list = pickle.load(f_data)

    f_Ini_y = open('Ini_y.txt', 'rb')
    Ini_y = pickle.load(f_Ini_y)

    f_y = open('y.txt', 'rb')
    y = pickle.load(f_y)

    x = [s.strip() for s in x_list]

    f_test_y_s = open('test_y.txt', 'rb')
    test_y = pickle.load(f_test_y_s)

    f_test_Ini_y_s = open('test_Ini_y.txt', 'rb')
    test_Ini_y = pickle.load(f_test_Ini_y_s)

    f_test_data_s = open('test_data.txt', 'rb')
    test_x_list = pickle.load(f_test_data_s)

    test_x = [test_s.strip() for test_s in test_x_list]

    return [x, y, Ini_y, test_x, test_y, test_Ini_y]


def pad_sentences(sentences, test_sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max([max(len(x) for x in sentences),max(len(tx) for tx in test_sentences)])

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

        test_padded_sentences = []
    for i in range(len(test_sentences)):
        test_sentence = test_sentences[i]
        test_num_padding = sequence_length - len(test_sentence)
        test_new_sentence = test_sentence + [padding_word] * test_num_padding
        test_padded_sentences.append(test_new_sentence)

    return padded_sentences, test_padded_sentences


def build_vocab(sentences, test_sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences,*test_sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, test_sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    x_test = np.array([[vocabulary[word] for word in sentence] for sentence in test_sentences])
    return [x, x_test]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, train_Ini_y, test_x, test_y, test_Ini_y = load_data_and_labels()
    sentences_padded, test_sentence_padded = pad_sentences(sentences, test_x)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded, test_sentence_padded)
    x, test_x = build_input_data(sentences_padded, test_sentence_padded, vocabulary)
    return [x, labels, test_x, test_y, vocabulary, vocabulary_inv, train_Ini_y, test_Ini_y]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
