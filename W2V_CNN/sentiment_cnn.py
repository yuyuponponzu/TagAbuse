"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
import data_helpers
from w2v import train_word2vec

from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
threshold = 50

# Training parameters
batch_size = 32
num_epochs = 250

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------

#lossfunc All[0]を避けるため
def lossfunc(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred)) + 0.01 * 1 / K.sum((y_pred)+K.epsilon())#precision

def P(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.20), 'float32'))

    precision = true_positives / (pred_positives + K.epsilon())
    return precision

#recall
def R(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.20), 'float32'))
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.20), 'float32'))

    recall = true_positives / (poss_positives + K.epsilon())
    return recall

#f-measure
def F(y_true, y_pred):
    p_val = P(y_true, y_pred)
    r_val = R(y_true, y_pred)
    f_val = 2*p_val*r_val / (p_val + r_val)

    return f_val

def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
    else:
        x, y, test_x, test_y, vocabulary, vocabulary_inv_list, train_Ini_y, test_Ini_y = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        #y = y.argmax(axis=1)

        num_output = train_Ini_y.shape[1]
        #初期タグ情報をくっつけれるようにresize
        q, mod = divmod(len(train_Ini_y[0]),embedding_dim)
        train_Ini_y = np.pad(train_Ini_y, [(0,0), (0,int(embedding_dim - mod))], "constant")
        test_Ini_y = np.pad(test_Ini_y, [(0,0), (0,int(embedding_dim - mod))], "constant")

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        train_x = x[shuffle_indices]
        train_y = y[shuffle_indices]
        train_Ini_y = train_Ini_y[shuffle_indices]

    return train_x, train_y, test_x, test_y, vocabulary_inv, train_Ini_y, test_Ini_y, q+1, num_output


# Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv, Ini_y_train, Ini_y_test, q, num_output = load_data(data_source)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim, min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        print(x_train.shape)
        reIni_y_train = np.reshape(Ini_y_train[:], (x_train.shape[0], q, embedding_dim))
        newx_train = np.empty((len(x_train), len(x_train[0])+len(reIni_y_train[0]), embedding_dim))
        for i in range(len(x_train)):
            newx_train[i] = np.concatenate([x_train[i], reIni_y_train[i]] , axis=0)
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        reIni_y_test = np.reshape(Ini_y_test[:], (x_test.shape[0], q, embedding_dim))
        newx_test = np.empty((len(x_test), len(x_test[0])+len(reIni_y_test[0]), embedding_dim))
        for i in range(len(x_test)):
            newx_test[i] = np.concatenate([x_test[i], reIni_y_test[i]] , axis=0)
        x_train = newx_train
        x_test = newx_test
        print("x_train static shape:", x_train.shape)
        print("x_test static shape:", x_test.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="sigmoid")(z)
model_output = Dense(num_output, activation="softmax")(z)

model = Model(model_input, model_output)
model.compile(loss= lossfunc, optimizer="adagrad", metrics=["binary_accuracy", P, R, F])

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

fold_num = 10
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True)
h = 0
#10バリデーション
log = open('./log.txt','w')

cat_y = []
for i in range(len(y_train)):
    s = [str(int(y_train[i][j])) for j in range(len(y_train[i]))]
    d = ''.join(s)
    cat_y.append(d)

for train, test in kfold.split(x_train,cat_y):
    h = str(int(h) + 1)

    # Train the model
    model.fit(x_train[train], y_train[train], batch_size=batch_size, epochs=num_epochs,
    validation_data=(x_train[test], y_train[test]), verbose=0)
    # Test the model with testdata
    scores = model.evaluate(x_test, y_test, verbose=0)
    pred = model.predict(x_test)
    report = classification_report(y_test, (pred>threshold).astype(int))

    #log 書き込み
    log.write(h+", scores:"+str(scores[1]*100)+"\n")
    log.write("report:"+report+"\n")
