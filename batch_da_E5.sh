#!/bin/bash
# データ拡張手法ごとにE5のファインチューニングでカテゴリ分類を一括で実行
for dat in GPT4DA SudachiDA wordNetDA BERT 
do
    python da_ML_E5.py ${dat}
done
