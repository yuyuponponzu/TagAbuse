import os
import re
import sys
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess as preprocess
from gensim.models import Doc2Vec



s ="""
"<p>I know this should be simple , but <code>android:scaleType=\"centerCrop\"</code> doesn't crop Image</p>&#xA;&#xA;<p>I got image <em>1950 pixels</em> wide and need it to be cropped by parent's width. But <code>android:scaleType=\"centerCrop\"</code> dosen't crop Image. What do I need to do in layout to show only first <em>400 pixels</em>, for instance or whatever screen/parent width is </p>&#xA;&#xA;<p>Sorry for simple question - tried to google it - only complicated questions there. And I'm new, so dont downvote plz)</p>&#xA;&#xA;<pre><code>&lt;RelativeLayout xmlns:android=\"http://schemas.android.com/apk/res/android\"&#xA;    xmlns:tools=\"http://schemas.android.com/tools\"&#xA;    android:id=\"@+id/rl1\"&#xA;    android:layout_width=\"match_parent\"&#xA;    android:layout_height=\"match_parent\"&#xA;    android:background=\"@color/background_color\"&gt;&#xA;&#xA;    &lt;ImageView&#xA;            android:id=\"@+id/ver_bottompanelprayer\"&#xA;            android:layout_width=\"match_parent\"&#xA;            android:layout_height=\"227px\"&#xA;            android:layout_alignParentBottom=\"true\"&#xA;            android:layout_alignParentLeft=\"true\"&#xA;            android:scaleType=\"matrix\"&#xA;&#xA;            android:background=\"@drawable/ver_bottom_panel_tiled_long\" /&gt;&#xA;&#xA;&lt;/RelativeLayout&gt;&#xA;</code></pre>&#xA;
"""

def splitQuestion(singlebody):
    Q = []
    code = []
    sepC = singlebody.split("<code>")
    p = re.compile(r"<[^>]*?>")
    Q.append(p.sub("",sepC[0]))

    for i in range(len(sepC)-1):
        sep_C = sepC[i+1].split("</code>")
        code.append(p.sub("",sep_C[0]))
        Q.append(p.sub("",sep_C[1]))

    print(Q)
    print(code)
    
    return Q, Code


Q, Code = splitQuestion(s)

flatQ = [item for sublist in Q for item in sublist]
flatCode = [item for sublist in Code for item in sublist]


import tqdm
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from libs.wakati import Wakati # 自作モジュール
 
# -----------------------------------------
# Function definition
# -----------------------------------------
 
def get_df(path):
    """csvファイルからDataFrameを取得する."""
    return pd.read_csv(path, encoding="utf-8")
 
def get_token_list(text_list):
    """テキストリストから分かち書きされた単語リストを取得する."""
    token_list = [Wakati(text).tokenize() for text in tqdm.tqdm(text_list)]
    return token_list
 
def get_d2v_documents(tag_list, token_list):
    """Doc2Vec学習用データを生成する."""
    documents = []
    for tag, token in zip(tag_list, token_list):
        documents.append(TaggedDocument(tags=[tag], words=token))
    return documents
 
def get_d2v_model(documents):
    """Doc2Vecで学習したモデルを取得する."""
    model = Doc2Vec(
        documents=documents,
        size=200,
        min_count=1,
        iter=5
    )
    return model
 
def get_df_docvecs(path):
    """各元素の200次元の分散表現(ベクトル)をDataFrameで取得する."""
    df = get_df(path)
    tag_list = list(df["SYMBOL"])
    text_list = list(df["TEXT"])
    token_list = get_token_list(text_list)
    documents = get_d2v_documents(tag_list, token_list)
    model = get_d2v_model(documents)
    df_docvecs = pd.DataFrame()
    for tag in tag_list:
        df_docvecs[tag] = model.docvecs[tag]
    df_docvecs = df_docvecs.T
    return df_docvecs
 
# ----------------------------------------
# Main processing
# ----------------------------------------
 
if __name__=="__main__":
    path = "extract_atoms_data.csv"
    df_docvecs = get_df_docvecs(path)
