import sys, os, re
import pandas as pd
import random
import wordnet_jp

# 乱数の固定
random.seed(100)

# 同義語の選択
def extSynonyms(word, num=5):
    synonym = wordnet_jp.getSynonym(word)
    wdL = []
    for k,v in synonym.items():
        for w in v:
            wdL.append(w)
    random.shuffle(wdL)

    return wdL[:num]


import csv

# データの読み込み
df = pd.read_csv('./data/ej_small_wakati.tsv', sep='\t')

# 拡張データの出力
wf = open('./data/ej_small_wordNetDA.tsv', 'w')
wf.write('ID\tSentence\tConverted\tIntent\n')
for i in range(len(df)):
    sen = df.iloc[i]['Jutterances']
    intent = df.iloc[i]['Intent']
    sa = sen.split(' ')
    s = re.sub(r'\s+', '', sen )
    rids = list(range(len(sa)))
    random.shuffle(rids)

    chg_flg = 0
    n = 0
    
    for j, ss in enumerate(sa):
        if n == 5:
            break
        if n >= len(rids):
            break

        if j == rids[n]:
            ids = extSynonyms(ss, 15)
            if len(ids) > 0:
                for syn in ids:
                    xa = sa.copy()
                    xa[j] = syn
                    print(sen, "====>", ' '.join(xa))
                    conv = ' '.join(xa)
                    wf.write(f'{i}\t{sen}\t{conv}\t{intent}\n')
                    n += 1

wf.close
