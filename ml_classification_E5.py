# BERT-DA, GPT-4-DA の2通りと
# Sudachi同義語辞書を用いた手法と比較
# 同義語等を使っているので，BERTをファインチューニングした分類器を作成する
# 学習データとテストデータは，8:2 の割合で分割する
#
import sys, os, re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import TrainingArguments, Trainer
from torch.optim import AdamW
import random
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


from torch import nn

def predict(text):
    inputs = tokenizer(text, add_special_tokens=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    ps = nn.Softmax(1)(outputs.logits)

    max_p = torch.max(ps)
    result = torch.argmax(ps).item()
    return result, max_p

# 変換関数
def preprocess_function(data):
    texts = [q.strip() for q in data["text"]]
    inputs = tokenizer(
        texts,
        max_length=32, 
        truncation=True,
        padding=True,
    )

    inputs['labels'] = torch.tensor(data['label'])

    return inputs


# デバイス判定
device = "cuda:0" if torch.cuda.is_available() else "cpu"
random.seed(100)

def makeDataSet(datapath):
    df = pd.read_csv(datapath, sep='\t')
    intents = list(set(list(df['Intent'].values)))
    id2label, label2id = {}, {}
    for i, lb in enumerate(intents):
        id2label[i] = lb
        label2id[lb] = i

    dic = {'text':[], 'label':[], 'id':[]}
    tData = {'text':[], 'label':[]}
    vData = {'text':[], 'label':[]}
    eData = {'text':[], 'label':[]}

    trains = {}
    for id, (intent, sen) in enumerate(df[['Intent', 'Jutterances']].values):
        sen = re.sub(r'\s+', '', sen)
        dic['text'].append(sen)
        dic['label'].append(label2id[intent])
        dic['id'].append(id)
        trains[id] = len(dic['text']) -1
    
    rids = list(trains.keys())
    # 0,1,2,3,4,5,6,7 ==> train
    # 8,9 ==> test
    random.shuffle(rids)
    n = 0
    
    print(len(rids), rids[0])

    check = {}
    for ri in rids:
        ii = trains[ri]
        if n % 8 != 0 and n % 9 != 0:            
            if n % 7 != 0:
                if len(tData['text']) >= 10000:
                    continue
                tData['text'].append(dic['text'][ii])
                tData['label'].append(dic['label'][ii])
            elif n % 7 == 0:
                vData['text'].append(dic['text'][ii])
                vData['label'].append(dic['label'][ii])
            
        else:
            eData['text'].append(dic['text'][ii])
            eData['label'].append(dic['label'][ii])
            #print("EDATA")
        n+=1

    tD = {'text': tData['text'], 'label': tData['label']}
    eD = {'text': eData['text'], 'label': eData['label']}
    
    return tD, eD, vData, id2label, label2id

model_name = "intfloat/multilingual-e5-base"
tData, eData, vData, id2label, label2id = makeDataSet(f'./data/ej_small_wakati.tsv')
print("Label Num: ", len(id2label))

tdf = pd.DataFrame(tData)
print(tdf.head())
edf = pd.DataFrame(eData)
print("==========\n", edf.head(), len(edf))
vdf = pd.DataFrame(vData)


ds_train = Dataset.from_pandas(tdf)
ds_valid = Dataset.from_pandas(vdf)
ds_test = Dataset.from_pandas(edf)

dataset = DatasetDict(
    { "train": ds_train,
     "validation": ds_valid,
     }
)
testset = DatasetDict(
    {"test": ds_test, }
)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(id2label)).to(device) #, id2label=id2label, label2id=label2id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outdir",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer, padding='max_length', max_length=32)

# 変換
tokenized_data = dataset.map(preprocess_function, batched=True)
tokenized_testdata = testset.map(preprocess_function, batched=True)

# 学習を実行
from sklearn.model_selection import train_test_split
from transformers import Trainer, EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

print("Training oK!")

evalRes = []
scores = []
for t in eData['text']:
    r, score = predict(t)
    print(t, "\t", r, "\t", score)
    evalRes.append(r)
    scores.append(score)

odir = './e5_res'
if not os.path.exists(odir):
    os.mkdir(odir)
    
wf = open(f'{odir}/result_wakati.tsv', 'w')
wf.write('Sentence\tAns.\tPred.\tScore\n')
for i, e in enumerate(evalRes):
    wf.write(f"{eData['text'][i]}\t{id2label[eData['label'][i]]}\t{id2label[e]}\t{scores[i]}\n")
wf.close
