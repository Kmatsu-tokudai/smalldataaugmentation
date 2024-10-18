#!/bin/bash
# データ拡張手法ごとにBERTのファインチューニングでカテゴリ分類を一括で実行
for dat in GPT4DA SudachiDA wordNetDA BERT  
do
    python da_ML_classification.py ${dat}
done
