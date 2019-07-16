import numpy as np
import re
import itertools
from collections import Counter
import csv
import math

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


def adjust_ratio(data, noc_data, diff, noc_y, Ini_y, noc_Ini_y, RATIO):
    I_ratio = len(data) / len(noc_data)
    count = RATIO / I_ratio
    if count != 1:
        cou_i = math.floor(count)
        new_data = data
        new_Ini_y = Ini_y
        new_diff = diff
        if count >= 1:
            for l in range(cou_i):
                new_data = np.append(new_data, data)
                new_Ini_y = np.append(new_Ini_y, Ini_y,axis=0)
                new_diff = np.append(new_diff, diff,axis=0)
        new_data = np.append(new_data, data[:int((count - cou_i) * len(data))])
        print(np.array(new_data).shape)
        new_Ini_y = np.append(new_Ini_y, Ini_y[:int((count - cou_i) * len(Ini_y))],axis=0)
        new_diff = np.append(new_diff, diff[:int((count - cou_i) * len(diff))],axis=0)
        new_i_ratio = len(new_data) / len(noc_data)
        new_count = RATIO / new_i_ratio

        data = np.append(new_data, noc_data)
        print(np.array(data).shape)
        Ini_y = np.append(new_Ini_y, noc_Ini_y, axis=0)
        diff = np.append(new_diff, noc_y, axis=0)

        if new_count != 1:
            #大体1に近づいておけば成功
            print("NEW_RATIO is ",new_i_ratio)
    
    return data, diff, Ini_y


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    type_delimiter = "\n|||||\n"
    
    # Load data from files
    RATIO = 1
    ab_question = list(open("./data/processed_docs.txt").readlines())
    ab_question = [s.strip() for s in ab_question]
    noc_question = list(open("./data/noc_processed_docs.txt").readlines())
    noc_question = [s.strip() for s in noc_question]
    noc_len = len(noc_question)

    #Load y from files
    f = open('./data/labels.txt', 'r').read().split('\n')
    ab_Y = []
    for i in range(len(f)):
        if len(f[i].split()) == 0:
            continue
        ab_Y.append(list(map(float, f[i].split())))
    ab_y = np.array(ab_Y, np.float32)
    print(ab_y.shape)

    #タグ最初!=最後のpostの最初につけられたタグ情報
    f = open('./data/I_labels.txt', 'r').read().split('\n')
    Ini_Y = []
    for i in range(len(f)):
        if len(f[i].split()) == 0:
            continue
        Ini_Y.append(list(map(float, f[i].split())))
    Ini_y = np.array(Ini_Y, np.float32)
    print(Ini_y.shape)
    #ラベルの差異を計算
    diff = np.abs(ab_y - Ini_y)
    # Generate labels
    noc_y = np.zeros((noc_len,diff.shape[1]))

    f = open('./data/noc_labels.txt', 'r').read().split('\n')
    noc_Ini_Y = []
    for i in range(len(f)):
        if len(f[i].split()) == 0:
            continue
        noc_Ini_Y.append(list(map(float, f[i].split()))) 
    noc_Ini_y = np.array(noc_Ini_Y, np.float32)
    #negative_examples = [s.strip() for s in negative_examples]
    question, y, Ini_y = adjust_ratio(ab_question, noc_question, diff, noc_y, Ini_y, noc_Ini_y, RATIO)

    test_question = list(open("./data/test_processed_docs.txt").readlines())
    test_question = [s.strip() for s in test_question]

    #Load y from files
    f = open('./data/test_labels.txt', 'r').read().split('\n')
    test_ab_y = []
    for i in range(len(f)):
        if len(f[i].split()) == 0:
            continue
        test_ab_y.append(list(map(float, f[i].split())))
    test_ab_y = np.array(test_ab_y, np.float32)
    print(test_ab_y.shape)

    #タグ最初!=最後のpostの最初につけられたタグ情報
    f = open('./data/test_I_labels.txt', 'r').read().split('\n')
    test_Ini_y = []
    for i in range(len(f)):
        if len(f[i].split()) == 0:
            continue
        test_Ini_y.append(list(map(float, f[i].split())))
    test_Ini_y = np.array(test_Ini_y, np.float32)
    print(test_Ini_y.shape)

    #ラベルの差異を計算
    y_test = np.abs(test_ab_y - test_Ini_y)

    # Split by words
    x = [s.split(" ") for s in question]
    x_test = [s.split(" ") for s in test_question]
    #Length of x_text is sentence list , and x_text[i] is word list.
    return [x, y, Ini_y, noc_len, x_test, y_test, test_Ini_y]


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


def build_input_data(sentences, labels, test_sentences, test_labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    x_test = np.array([[vocabulary[word] for word in sentence] for sentence in test_sentences])
    return [x, labels, x_test, test_labels]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, I_labels, noc_len, x_test, y_test, test_Ini_y = load_data_and_labels()
    sentences_padded, test_sentence_padded = pad_sentences(sentences, x_test)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded, test_sentence_padded)
    x, y, x_test, y_test = build_input_data(sentences_padded, labels, test_sentence_padded, y_test, vocabulary)
    return [x, y, x_test, y_test, vocabulary, vocabulary_inv, I_labels, test_Ini_y, noc_len]


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
