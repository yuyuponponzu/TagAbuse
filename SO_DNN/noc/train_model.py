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

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np

ktf.set_session(tf.InteractiveSession())
#label（one hot 表記）を開く
f = open('./labels.txt', 'r').read().split('\n')
Y = []
for i in range(len(f)):
    if len(f[i].split()) == 0:
        continue
    Y.append(list(map(float, f[i].split())))
y = np.array(Y, np.float32)
print(y.shape)

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

noc_data = open('./noc_processed_docs.txt', 'r').read().split('\n')[:-1]
vectorizer = TfidfVectorizer(input='content', token_pattern=r'\S+', analyzer='word', min_df=0.0002)
noc_y = np.zeros((len(noc_X),diff.shape[1]))

print("diff:",diff.shape)
print("noc_data:",noc_data.shape)
data = open('./processed_docs.txt', 'r').read().split('\n')[:-1]
print("data:",data.shape)
Ini_y = np.append(Ini_y, noc_y)
data = np.append(data, noc_data)
print("new_data:",data.shape)
while(1):pass

X = vectorizer.fit_transform(data).todense()
print(X.shape)
X = np.c_[Ini_y,X]
print(X.shape)
num_tags = y.shape[1]
features = X.shape[1]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

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
for i in range(len(y)):
    s = [str(int(y[i][j])) for j in range(len(y[i]))]
    d = ''.join(s)
    cat_y.append(d)
#10バリデーション
for train, test in kfold.split(X,cat_y):
    i = int(i) + 1
    i = str(i)
    model = Sequential()
    model.add(Dense(10500, activation='linear', input_dim=features))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dense(2048, activation='linear'))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dense(512, activation='linear'))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dense(256, activation='linear'))
    model.add(LeakyReLU(alpha=.1))
    model.add(Dense(num_tags, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X[train], diff[train], epochs=1, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping])
    model.save('./model_10val/'+i+'.h5')
    print("Model trained and saved")

    #model = load_model('./model_10val/3122.h5')
    scores = model.evaluate(X[test], diff[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    pred = model.predict(X[test])
    report = classification_report(diff[test], (pred>threshold).astype(int))
    log.write(i+"\n")
    log.write("scores:"+str(scores[1]*100)+"\n")
    log.write("report:"+report+"\n")
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
