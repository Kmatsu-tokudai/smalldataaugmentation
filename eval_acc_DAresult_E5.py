#
import sys, os, re, glob
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

if not os.path.exists('./sumE5'):
    os.mkdir('./sumE5')

for datype in ['GPT4DA', 'SudachiDA', 'wordNetDA', 'BERT']:
    wf = open(f'./sumE5/summary_{datype}.txt', 'w')
    df = pd.read_csv(f'e5_res/result_{datype}.tsv', sep='\t')
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
