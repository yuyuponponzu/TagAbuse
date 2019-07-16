import os
import re
import sys
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess as preprocess
from gensim.models import Doc2Vec
from bs4 import BeautifulSoup, Comment


s ="""
"<p>I know this should be simple , but <code>android:scaleType=\"centerCrop\"</code> doesn't crop Image</p>&#xA;&#xA;<p>I got image <em>1950 pixels</em> wide and need it to be cropped by parent's width. But <code>android:scaleType=\"centerCrop\"</code> dosen't crop Image. What do I need to do in layout to show only first <em>400 pixels</em>, for instance or whatever screen/parent width is </p>&#xA;&#xA;<p>Sorry for simple question - tried to google it - only complicated questions there. And I'm new, so dont downvote plz)</p>&#xA;&#xA;<pre><code>&lt;RelativeLayout xmlns:android=\"http://schemas.android.com/apk/res/android\"&#xA;    xmlns:tools=\"http://schemas.android.com/tools\"&#xA;    android:id=\"@+id/rl1\"&#xA;    android:layout_width=\"match_parent\"&#xA;    android:layout_height=\"match_parent\"&#xA;    android:background=\"@color/background_color\"&gt;&#xA;&#xA;    &lt;ImageView&#xA;            android:id=\"@+id/ver_bottompanelprayer\"&#xA;            android:layout_width=\"match_parent\"&#xA;            android:layout_height=\"227px\"&#xA;            android:layout_alignParentBottom=\"true\"&#xA;            android:layout_alignParentLeft=\"true\"&#xA;            android:scaleType=\"matrix\"&#xA;&#xA;            android:background=\"@drawable/ver_bottom_panel_tiled_long\" /&gt;&#xA;&#xA;&lt;/RelativeLayout&gt;&#xA;</code></pre>&#xA;
"""
"""
def splitQuestion(singlebody):
    Q = []
    code = []
    sepC = singlebody.split("<code>")
    p = re.compile(r"<[^>]*?>")
    p.strip("$#xA")
    p.strip("\n")
    Q.append(p.sub("",sepC[0]))

    for i in range(len(sepC)-1):
        sep_C = sepC[i+1].split("</code>")
        code.append(p.sub("",sep_C[0]))
        Q.append(p.sub("",sep_C[1]))

    
    return Q, code
"""


soup = BeautifulSoup(s, "lxml")
for text in soup.find_all(text=True):
    if text.strip():
        print(text)
        
while(1):pass

Q, code = splitQuestion(s)

flatQ = '&#xA;\n'.join(Q)
flatCode = '&#xA;\n'.join(code)
# 学習データ読み込み
train_data = [flatQ]
#assert train_data != ...
train_corpus = [flatQ]


train_corpus = [
    TaggedDocument(preprocess(doc), [i])
    for i, doc in enumerate(train_data)]
# モデル作成
model = Doc2Vec(size=200)
model.build_vocab(train_corpus)
# 学習
model.train(train_corpus, total_examples=model.corpus_count, epochs=10)

print(model.infer_vector(preprocess("This is a true.")))
