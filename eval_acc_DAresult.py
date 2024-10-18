# データ拡張手法ごとのBERTファインチューニングカテゴリ分類の評価
import sys, os, re, glob
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

if not os.path.exists('./sum'):
    os.mkdir('./sum')

for datype in ['GPT4DA', 'SudachiDA', 'wordNetDA', 'BERT']:
    wf = open(f'./sum/summary_{datype}.txt', 'w')
    df = pd.read_csv(f'bert_res/result_{datype}.tsv', sep='\t')
    lbs = list(set(df['Ans.'].values))
    lbs += list(set(df['Pred.'].values))
    lbs = list(set(lbs))
    print(len(lbs))
    ans, out = [], []
    for l in df['Ans.'].values:
        ans.append(lbs.index(l))
    for l in df['Pred.'].values:
        out.append(lbs.index(l))
    
    wf.write(f"{accuracy_score(ans, out)}\n")
    wf.write(f"{classification_report(ans, out, target_names=lbs)}\n")
    wf.close
