# smalldataaugmentation
## File構成
```
small_data_augmentation
├── README.txt (このファイル）
├── batch_da.sh　（データ拡張手法ごとにBERTのファインチューニングでカテゴリ分類を一括で実行）
├── batch_da_E5.sh（データ拡張手法ごとにE5のファインチューニングでカテゴリ分類を一括で実行）
├── bert_da.py　（BERTを用いたデータ拡張）
├── da_ML_E5.py　（E5のファインチューニングによるカテゴリ分類）
├── da_ML_classification.py　（BERTのファインチューニングによるカテゴリ分類）
├── da_gpt4o.py　（GPT-4oによるデータ拡張）
├── data
│   ├── ej_small_wakati_DA.tsv (英日翻訳済み分かち書きテキストデータ・発話意図ラベル付き）/*一部誤(未)翻訳*/
│   ├── synonyms.txt　（SudachiPy同義語辞書）
│   └── synonyms_base.csv 　（SudachiPy同義語辞書からIDと単語のみ抽出したCSVファイル）
├── eval_acc_DAresult.py （データ拡張手法ごとのBERTファインチューニングカテゴリ分類の評価）
├── eval_acc_DAresult_E5.py　（データ拡張手法ごとのE5ファインチューニングカテゴリ分類の評価）
├── eval_acc_withoutDAresult.py   (データ拡張無しのファインチューニングカテゴリ分類の評価）
├── ml_classification.py （データ拡張無しのBERTファインチューニングカテゴリ分類の評価）
├── ml_classification_E5.py （データ拡張無しのE5ファインチューニングカテゴリ分類の評価）
├── sudachi_DA.py　（SudachiPy辞書によるデータ拡張）
├── wn_DA.py　　（日本語WordNetによるデータ拡張）
├── wnjpn.db　（日本語WordNetのデータベースファイル）/*Not here*/
└── wordnet_jp.py　（日本語WordNetのデータベースを扱うためのライブラリ）
```

## Datasets
Small Data Augmentation (https://www.kaggle.com/datasets/salmanfaroz/small-talk-intent-classification-data)
このテキストデータにおける文章を、機械翻訳して日本語にしたものを用いる。ラベルはそのまま用いる。
（一部翻訳ミスあり）

## Small Data Augmentation
※データ拡張に成功したデータのみを拡張データとする．（拡張データ：ID,Converted,Sentence,Intent）
### GPT4o
以下の箇所に，取得したAPI Keyに変更
```python:da_gpt4o.py
~
##########################################
# 取得したAPI Keyを以下に代入する
openai.api_key="OPENAI_API_KEY"
##########################################
~
```
#### 実行環境
```
python 3.10.9
openai 1.51.2
pandas 2.0.3
natto-py 1.0.1
mecab 0.996.5
```
### BERT
### sudachi
### 日本語WordNet
#### 実行環境
```
python 3.7
pandas 1.1.5
transformers 4.30.2
SudachiPy 0.6.7
```
## カテゴリ分類
拡張データのみでカテゴリ分類を行う．（withoutは，元の英日翻訳データセットを用いる）
### without(accuracy)
||acc|DataSize|
|:---|---:|---:|
|BERT|0.23|2596|
|E5|0.67|2596|

### BERT
#### Results(accuracy)
||acc|DataSize|
|:---|---:|---:|
|GPT4o|0.39|24757|
|BERT|0.52|37801|
|Sudachi|0.15|500|
|WordNet|0.43|7730|

### E5
#### Results(accuracy)
||acc|DataSize|
|:---|---:|---:|
|GPT4o|0.65|24757|
|BERT|0.73|37801|
|Sudachi|0.60|500|
|WordNet|0.62|7730|
