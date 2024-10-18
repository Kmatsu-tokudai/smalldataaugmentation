import sys, os, re
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM

# BERTの日本語モデル
model_name = 'tohoku-nlp/bert-base-japanese'

# tokenizerとmodelの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
# pipelineの設定
mask_filter = pipeline("fill-mask", model=model_name)

# データの読み込み
df = pd.read_csv('./data/ej_small_wakati.tsv', sep='\t')
i = 0
N = 10 # 10テキスト毎で拡張する

import random
random.seed(100) # 乱数の固定

# 拡張データの出力
wf = open(f'./data/ej_small_BERT.tsv', 'w')
wf.write('ID\tSentence\tMaskedSentence\tConverted\tIntent\n')
while i+N < len(df):
    n = i
    txts, labs, sents, ids = [], [], [], []
    while n < i+N:
        # ランダムに１単語をMASKしたテキストの生成
        for c in range(3):
            sa = re.split(r'\s+', df.iloc[n]['Jutterances'])
            sai = list(range(len(sa)))
            random.shuffle(sai)
            if len(sai) <= c:
                continue
            sents.append(' '.join(sa))
            sa[sai[c]] = '[MASK]'
            txts.append(''.join(sa) )
            labs.append(df.iloc[n]['Intent'])
            ids.append(n)

        n+=1
    
    # MASKに対してBERTの予測単語候補の上位5単語を用いてテキストを生成
    rr = mask_filter(txts, top_k = 5)
    x = i
    for j in range(len(rr)):
        ss = ''
        for k in range(len(rr[j])):
            ss = rr[j][k]['sequence']
            wf.write(f"{ids[j]}\t{sents[j]}\t{txts[j]}\t{ss}\t{labs[j]}\n")
        print(ss)
        x += 1
    i+=N

# 拡張したデータ数が元のデータ数に満たないとき，残りのデータ拡張を行う
# テキストが3単語未満時に，エラーが出るかも
if i < len(df):
    n = i
    txts, labs, sents, ids = [], [], [], []
    while n < len(df):
        for c in range(3):
            sa = re.split(r'\s+', df.iloc[n]['Jutterances'])
            sai = list(range(len(sa)))
            random.shuffle(sai)
            if len(sai) <= c:
                continue
            sents.append(' '.join(sa))
            sa[sai[c]] = '[MASK]'
            txts.append(''.join(sa) )
            labs.append(df.iloc[n]['Intent'])
            ids.append(n)
        n+=1
    rr = mask_filter(txts, top_k = 5)
    x = i
    for j in range(len(rr)):
        ss = ''
        for k in range(len(rr[j])):
            ss = rr[j][k]['sequence']
            wf.write(f"{ids[j]}\t{sents[j]}\t{txts[j]}\t{ss}\t{labs[j]}\n")
        print(ss)
        x += 1

wf.close
