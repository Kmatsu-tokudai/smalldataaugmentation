import sys, os, re
import pandas as pd
from sudachipy import tokenizer
from sudachipy import dictionary
import random

import csv

# 同義語データベースの構築
def mk_synonymDB():
    # 同義語データの読み込み
    with open("./data/synonyms.txt", "r") as f:
        reader = csv.reader(f)
        data = [r for r in reader]

    output_data = []
    synonym_set = []
    synonym_group_id = None
    for line in data:
        if not line:
            if synonym_group_id:
                base_keyword = synonym_set[0]
                output_data.append([
                    synonym_group_id, base_keyword
                ])
            synonym_set = []
            continue
        else:
            synonym_group_id = line[0]
            synonym_set.append(line[8])

    # 単語データの出力
    with open("./data/synonyms_base.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(output_data)

# 単語データの読み込み
with open('./data/synonyms_base.csv', "r") as f:
    reader = csv.reader(f)
    data = [[int(r[0]), r[1]] for r in reader]
    synonyms = dict(data)

# 乱数の固定
random.seed(100)

# tokenizerの読み込みとmodeの指定
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.B

# データの読み込み
df = pd.read_csv('./data/ej_small_wakati.tsv', sep='\t')

# 拡張データの出力
wf = open('./data/ej_small_SudachiDA.tsv', 'w')
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

        if j == rids[n]:
            token = tokenizer_obj.tokenize(ss, mode)[0]
            ids = token.synonym_group_ids()
            if len(ids) > 0:
                for sid in ids:
                    surf = synonyms[sid]
                    #print(ss, "==>", surf)
                    xa = sa.copy()
                    xa[j] = surf
                    print(sen, "====>", ' '.join(xa))
                    conv = ' '.join(xa)
                    wf.write(f'{i}\t{sen}\t{conv}\t{intent}\n')
                    n += 1

wf.close
