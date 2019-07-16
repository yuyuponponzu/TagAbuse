# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np
import math
import matplotlib.pyplot as plt
#変更されたデータと変更されていないデータの割合
RATIO = 1
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
ktf.set_session(tf.InteractiveSession())

#タグ最初!=最後のpostの最後につけられたタグ情報,label（one hot 表記）を開く
f = open('./labels.txt', 'r').read().split('\n')
Y = []
for i in range(len(f)):
    if len(f[i].split()) == 0:
        continue
    Y.append(list(map(float, f[i].split())))
y = np.array(Y, np.float32)
print(y.shape)

#タグ最初!=最後のpostの最初につけられたタグ情報
f = open('./I_labels.txt', 'r').read().split('\n')
Ini_Y = []
for i in range(len(f)):
    if len(f[i].split()) == 0:
        continue
    Ini_Y.append(list(map(float, f[i].split())))
Ini_y = np.array(Ini_Y, np.float32)
print(Ini_y.shape)

#ラベルの差異を計算
diff = np.abs(y - Ini_y)

#タグ最初=最後のpostは差異がないので全て0を立てる
noc_data = open('./noc_processed_docs.txt', 'r').read().split('\n')[:-1]
vectorizer = TfidfVectorizer(input='content', token_pattern=r'\S+', analyzer='word', min_df=0.0002)
noc_y = np.zeros((len(noc_data),diff.shape[1]))
print(np.array(noc_y).shape)

#タグ最初=最後のpostについてのタグ情報
f = open('./noc_labels.txt', 'r').read().split('\n')
noc_Ini_Y = []
for i in range(len(f)):
    if len(f[i].split()) == 0:
        continue
    noc_Ini_Y.append(list(map(float, f[i].split())))
noc_Ini_y = np.array(noc_Ini_Y, np.float32)
print(np.array(noc_Ini_y).shape)

data = open('./processed_docs.txt', 'r').read().split('\n')[:-1]

"""
data = list(data)
diff = list(diff)
Ini_y = list(Ini_y)
noc_data = list(noc_data)
noc_y = list(noc_y)
noc_Ini_y = list(noc_Ini_y)
"""


test_data = []
test_diff = []
test_Ini_y = []
test_noc_data = []
test_noc_y = [] 
test_noc_Ini_y = []


print(np.array(data).shape,diff.shape,np.array(noc_data).shape,noc_y.shape)
for j in range(1000):
    test_data.append(data[j])
    test_diff.append(diff[j])
    test_Ini_y.append(Ini_y[j])
    test_noc_data.append(noc_data[j])
    test_noc_y.append(noc_y[j])
    test_noc_Ini_y.append(noc_Ini_y[j])
    np.delete(data, j, 0)
    np.delete(diff, j, 0)
    np.delete(Ini_y, j, 0)
    np.delete(noc_data, j, 0)
    np.delete(noc_y, j, 0)
    np.delete(noc_Ini_y, j, 0)

test_data = np.array(test_data)
test_diff = np.array(test_diff)
test_Ini_y = np.array(test_Ini_y)
test_noc_data = np.array(test_noc_data)
test_noc_y = np.array(test_noc_y)
test_noc_Ini_y = np.array(test_noc_Ini_y)
print(test_data.shape,test_diff.shape,test_Ini_y.shape,test_noc_data.shape,test_noc_y.shape,test_noc_Ini_y.shape)
print(np.array(data).shape,np.array(diff).shape,np.array(noc_data).shape,np.array(noc_y).shape)

test_data_s = []
test_data_s.append(test_data)
test_data_s.append(test_noc_data)
test_Ini_y_s = []
test_Ini_y_s.append(test_Ini_y)
test_Ini_y_s.append(test_noc_Ini_y)
test_Ini_y_s = np.array(test_Ini_y_s).reshape([-1,np.array(test_Ini_y_s).shape[2]])
test_y_s = []
test_y_s.append(test_diff)
test_y_s.append(test_noc_y)
test_y_s = np.array(test_y_s).reshape([-1,np.array(test_y_s).shape[2]])

print(np.array(test_data_s).shape, np.array(test_Ini_y_s).shape, np.array(test_y_s).shape)

"""
#予め設定した重みになるようにタグ最初!=最後のpostのデータを水増しする
I_ratio = len(data) / len(noc_data)
count = RATIO / I_ratio
if count != RATIO:
    cou_i = math.floor(count)
    new_data = data
    print(np.array(new_data).shape)
    new_Ini_y = Ini_y
    new_diff = diff
    if count >= RATIO:
        for l in range(cou_i):
            new_data = np.append(new_data, data)
            print(np.array(new_data).shape)
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

    if new_count != RATIO:
        #大体RATIOに近づいておけば成功
        print("NEW_RATIO is ",new_i_ratio)
"""
data = np.append(new_data, noc_data)
print(np.array(data).shape)
Ini_y = np.append(new_Ini_y, noc_Ini_y, axis=0)
diff = np.append(new_diff, noc_y, axis=0)


print("data",np.array(data).shape)
print("Ini_y",np.array(Ini_y).shape)

X = vectorizer.fit_transform(data).todense()
print(X.shape)
X = np.c_[Ini_y,X]
print(X.shape)
num_tags = y.shape[1]
features = X.shape[1]

"""
#testデータを同様にして準備
f = open('./test_labels.txt', 'r').read().split('\n')
test_Y = []
for i in range(len(f)):
    if len(f[i].split()) == 0:
        continue
    test_Y.append(list(map(float, f[i].split())))
test_y = np.array(test_Y, np.float32)
print(test_y.shape)

f = open('./test_I_labels.txt', 'r').read().split('\n')
test_Ini_Y = []
for i in range(len(f)):
    if len(f[i].split()) == 0:
        continue
    test_Ini_Y.append(list(map(float, f[i].split())))
test_Ini_y = np.array(test_Ini_Y, np.float32)
print(test_Ini_y.shape)

#ラベルの差異を計算
test_diff = np.abs(test_y - test_Ini_y)

test_data = open('./test_processed_docs.txt', 'r').read().split('\n')[:-1]
test_X = vectorizer.transform(test_data).todense()
print(test_X.shape)
"""
test_data_s = vectorizer.transform(np.array(test_data_s).reshape([-1])).todense()
test_X = np.c_[np.array(test_Ini_y_s),test_data_s]
test_diff = np.array(test_y_s)


