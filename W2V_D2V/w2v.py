import os
import re
import sys
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess as preprocess
from gensim.models import Doc2Vec
import json
#Json読み込み
f = open('10815posts_score5.json', 'r')
jsonData = json.load(f)
f.close()

#&~~~;みたいなタグを全て消す
begin = "&"
end = ";"

def splitQuestion(singlebody):
    Q = []
    code = []
    singlebody.strip("\n")
    singlebody.strip('"')
    sepC = singlebody.split("<code>")
    singlebody = re.sub('(%s.*%s)' % (begin,end), '', singlebody)
    #<>でcode部分を切り出す
    p = re.compile(r"<[^>]*?>")
    Q.append(p.sub("",sepC[0]))
    for i in range(len(sepC)-1):
        sep_C = sepC[i+1].split("</code>")
        code.append(p.sub("",sep_C[0]))
        #コード文の後に質問文が来ない時があるから，それを回避
        if len(sep_C) != 1:
            Q.append(p.sub("",sep_C[1]))
    return Q, code

def splitTag(singletag):
    singletag = singletag.split(">")
    sepT = [key.split("<")[1] for key in singletag[:-1]]
    return sepT

Qs = []
codes = []
tags = []
for i, Data in enumerate(jsonData):
    #各ポストのデータ読み込み
    #sにはタイトル＋文章全てが入ってる
    s = Data['Title'] + Data['Body']
    #tagはtagsに全ての分が順番に格納されている（複数保持）
    tag = Data['Tags']
    #デフォルトではtagは<-><-><->と繋がってるから外してリスト化
    tag = splitTag(tag)
    #score = Data['Score']
    Q, code = splitQuestion(s)
    flatQ = '\n'.join(Q)
    flatCode = '\n'.join(code)
    flatQ = re.sub("&#xA;", " ", flatQ)
    flatCode = re.sub("&#xA;", " ", flatCode)
    #Qsに全ての質問文章データ（今回であれば10815個のポスト分）が格納されている
    Qs.append(flatQ)
    #codesに全てのコードデータ（今回であれば10815個のポスト分）が格納されている．
    codes.append(flatCode)
    tags.append(tag)
"""preprocess(doc)で単語ごとの分ち書きに変えてる．この場合のaはデータ数分だけの配列を持ち，各配列に対して分ち書きした単語が一つずつ保持されている
a = [preprocess(doc) for doc in (Qs)]
print(a)
"""

print(tags)
print(len(tags))
train_qs_corpus = [
    TaggedDocument(preprocess(doc), tags[i])
    for i, doc in enumerate(Qs)]

"""Codesに対するモデル
train_codes_corpus = [
    TaggedDocument(preprocess(codes), [i])
    for i, doc in enumerate(train_data)]

modelCodes = Doc2Vec(size=200)
model.build_vocab(train_codes_corpus)
# 学習
model.train(train_codes_corpus, total_examples=model.corpus_count, epochs=10)
"""

# モデル作成(とりあえずデフォルト値でのモデル作成は以下)
#model = Doc2Vec(size=200,dm=1)

#qiitaにてうまくいってそうだったモデルは以下(https://qiita.com/naotaka1128/items/2c4551abfd40e43b0146)
# alpha: 学習率 / min_count: X回未満しか出てこない単語は無視
# size: ベクトルの次元数 / iter: 反復回数 / workers: 並列実行数
model = Doc2Vec(alpha=0.025, min_count=5,
                       size=200, dm=1, iter=20, workers=4)
#単語の辞書を作ってる(one hot vectorの為)
model.build_vocab(train_qs_corpus)

# 学習
model.train(train_qs_corpus, total_examples=model.corpus_count, epochs=200)
# モデルセーブ
model.save('./data/doc2vec.model')

# 学習後はモデルをファイルからロード可能
#model = Doc2Vec.load('./data/doc2vec.model')
#print(model.most_similar("C"))
