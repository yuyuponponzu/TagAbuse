from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Activation
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
#変更されたデータと変更されていないデータの割合
RATIO = 1

ktf.set_session(tf.InteractiveSession())

"""
#ミニバッチ単位でのprecisionの平均
def P(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.2), 'float32'), axis=0)
    pred_positives = K.sum(K.cast(K.greater(K.clip(y_pred, 0, 1), 0.2), 'float32'), axis=0)

    precision = K.mean(true_positives / (pred_positives + K.epsilon()) )
    return precision

#ミニバッチ単位でのrecallの平均
def R(y_true, y_pred):
    true_positives = K.sum(K.cast(K.greater(K.clip(y_true * y_pred, 0, 1), 0.2), 'float32'), axis=0)
    poss_positives = K.sum(K.cast(K.greater(K.clip(y_true, 0, 1), 0.2), 'float32'), axis=0)

    recall = K.mean(true_positives / (poss_positives + K.epsilon()))
    return recall

#ミニバッチ単位でのf-measureの平均
def F(y_true, y_pred):
    p_val = P(y_true, y_pred)
    r_val = R(y_true, y_pred)
    f_val = 2*p_val*r_val / (p_val + r_val)
    return f_val
"""

#precision
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

def lossfunc(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred)) + 0.01 * 1 / K.sum((y_pred)+K.epsilon())# - 0.1 * K.tanh(P(y_true,y_pred))
# + 0.1 * K.tanh(P(y_true, y_pred)) + 0.1 * K.tanh(R(y_true, y_pred))
#+ 0.01 * 1 / K.sum((y_pred)) + 0.1 * K.tanh(P(y_true,y_pred))
#+ 1/(P(y_true, y_pred) + K.epsilon()) + 1/(R(y_true, y_pred) + K.epsilon()) 
#10000 *K.mean(K.binary_crossentropy(y_true, y_pred), axis=1) + 0.01 /(K.sum(y_pred, axis=1) + K.epsilon())
# 100 * K.sum(1 /(y_pred + K.epsilon()),axis=1) #+ 1/(P(y_true, y_pred) + K.epsilon())
#+ 1/(P(y_true, y_pred)+0.15) + 1/(R(y_true, y_pred)+0.15) 
#+ 1 / K.sum(y_pred , axis=0) , axis=-1)

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

#予め設定した重みになるようにタグ最初!=最後のpostのデータを水増しする
I_ratio = len(data) / len(noc_data)
count = RATIO / I_ratio
if count != 1:
    cou_i = math.floor(count)
    new_data = data
    print(np.array(new_data).shape)
    new_Ini_y = Ini_y
    new_diff = diff
    if count >= 1:
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

    if new_count != 1:
        #大体1に近づいておけば成功
        print("NEW_RATIO is ",new_i_ratio)

print("data",np.array(data).shape)
print("Ini_y",np.array(Ini_y).shape)

X = vectorizer.fit_transform(data).todense()
print(X.shape)
X = np.c_[Ini_y,X]
print(X.shape)
num_tags = y.shape[1]
features = X.shape[1]

fold_num = 10
i = 0
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True)
cvscores = []
macthscores = []
threshold = 0.1
validation_split = 0.25
early_stopping = EarlyStopping(patience=0, verbose=1)
#kfold.splitが二次元配列を受け取れないから，元のコードを弄るのは嫌なので無理やり１次元配列化してる
cat_y = []
log = open('./log.txt','w')
for i in range(len(diff)):
    s = [str(int(diff[i][j])) for j in range(len(diff[i]))]
    d = ''.join(s)
    cat_y.append(d)

#10バリデーション
for train, test in kfold.split(X,cat_y):
    h = int(h) + 1
    h = str(h)

    model = Sequential()
    model.add(Dense(10240, activation='linear', input_dim=features))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dropout(0.2))
    model.add(Dense(5196, activation='linear'))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dropout(0.2))
    model.add(Dense(2048, activation='linear'))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='linear'))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='linear'))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dropout(0.2))
    model.add(Dense(num_tags, activation='sigmoid'))

    model.compile(optimizer='Adam', loss = lossfunc, metrics=['binary_accuracy', P, R, F])
    model.fit(X[train], diff[train], epochs=200, batch_size=32, verbose=1)#, validation_split=0.1, callbacks=[early_stopping])
    model.save('./model_10val/'+h+'.h5')
    print("Model trained and saved")

    #model = load_model('./model_10val/3122.h5')
    scores = model.evaluate(X[test], diff[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    pred = model.predict(X[test])
    print(pred.shape)
    report = classification_report(diff[test], (pred>threshold).astype(int))
    log.write(h+"\n")
    log.write("scores:"+str(scores[1]*100)+"\n")
    log.write("report:"+report+"\n")

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

