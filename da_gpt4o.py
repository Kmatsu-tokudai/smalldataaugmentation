# OpenAIのAPIを用いてデータの拡張を行う
import openai
# MeCabの使用のためnatto-pyをインストールしておく
from natto import MeCab
nm = MeCab()

##########################################
# 取得したAPI Keyを以下に代入する
openai.api_key="OPENAI_API_KEY"
##########################################

import pandas as pd
import re

def GPT4DA():
	# データの読み込み
	df = pd.read_csv('./data/ej_small_wakati.tsv', sep='\t')
	# 拡張データの出力
	of = open('./data/ej_small_wakati_ID.tsv', 'w')
	wf = open('./data/ej_small_wakati_DA.tsv', 'w')
	of.write('ID\tSentence\tIntent\n')
	wf.write('ID\tSentence\tIntent\n')
	for i in range(len(df)):
	    u = df.iloc[i]['Jutterances']
	    intent = df.iloc[i]['Intent']
	    of.write(f'{i}\t{u}\t{intent}\n')
	    us = re.sub(r'\s+', '', u)
	    # モデルとプロンプトの設定
	    response = openai.chat.completions.create(
	    model="gpt-4o",
	    messages=[
	        {"role": "system", "content": "あなたは日本語のネイティブです。次の日本語文を意味を変えずに色々な言い回しで言い換えてください。言い換え文は全部で10通り作成してください。言い換え文以外の余計な文章は付け加えないでください。"},
	        {"role": "system", "content": us}
	    ]
	    )
	    try:
	        jsen = response.choices[0].message.content
	    except:
	        print(f"ID: {i}, OpenAI API Error!")
	        continue
	    
	    jsen = re.sub(r'^\n+', '', jsen)
	    jsen = re.sub(r'\n+$', '', jsen)
	    jsen = re.sub(r'^\s+', '', jsen)
	    jsen = re.sub(r'\s+$', '', jsen)
	    #print(jsen)
	    # 生成時の先頭の数字の削除
	    for js in jsen.split('\n'):
	        sen = re.sub(r'^[0-9]+\.\s+', '', js)
	        ps = nm.parse(sen)
	        words = []
	        for ws in ps.split('\n'):
	            wa = re.split(r'\t+', ws)
	            words.append(wa[0])
	        wkt = ' '.join(words)
	        wf.write(f"{i}\t{wkt}\t{intent}\n")
	    print(jsen, "... DataAug OK")

# GPT4DAの作成
def createGPT4DA() :
	df_id = pd.read_csv('./data/ej_small_wakati_ID.tsv', sep='\t')
	df_da = pd.read_csv('./data/ej_small_wakati_DA.tsv', sep='\t')
	wf = open(f'./data/ej_small_GPT4DA.tsv', 'w')
	wf.write('ID\tConverted\tSentence\tIntent\n')
	for ID, Sentence, Intent in df_da.values :
	    Converted = df_id.iloc[ID]['Sentence']
	    print(ID, Converted, re.sub(' EOS', '', Sentence), Intent)
	    wf.write(f"{ID}\t{Converted}\t{re.sub(' EOS', '', Sentence)}\t{Intent}\n")

# gpt4o 英日翻訳sample
def convertEtoJ():
	# 翻訳テキスト対の出力
	wf = open(f'./ej_small_data_da.tsv', 'w')
	wf.write('Intent\tEutterances\tJutterances\n')
	nc = 0
	for u,i in uh.items():
	    # モデルとプロンプトの設定
	    response = openai.chat.completions.create(
	    model="gpt-4o",
	    messages=[
	        {"role": "system", "content": "あなたはプロの優秀な日本語と英語の通訳者です。英語の入力を日本語に翻訳してください。日本語の入力を英語に翻訳してください。質問の入力があった場合は質問を翻訳してください。翻訳には丁寧な表現を使ってください。翻訳以外の言葉は付け加えないでください。"},
	        {"role": "system", "content": u}
	    ]
	    )
	    jsen = response.choices[0].message.content
	    jsen = re.sub(r'^\n+', '', jsen)
	    jsen = re.sub(r'\n+$', '', jsen)
	    jsen = re.sub(r'^\s+', '', jsen)
	    jsen = re.sub(r'\s+$', '', jsen)
	    wf.write(f'{i}\t{u}\t{jsen}\n')
	    if nc % 100 == 0:
	        print(jsen)
	    
	    nc += 1
	    
	    
if __name__ == '__main__':
	GPT4DA()
	createGPT4DA()