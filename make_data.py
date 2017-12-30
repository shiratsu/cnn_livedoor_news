# -*- coding: utf-8 -*-

import tensorflow as tf
import MeCab
m = MeCab.Tagger ("-Ochasen")

words = {}

def tokenize(text,ary):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    node = mecab.parseToNode(text)
    while node:
        ary.append(node.text)        
    return ary

# ファイルを読む
def readfile(dirname,filename):
    fh = open('text/'+dirname+'/'+filename, "r", encoding="UTF-8")
    wordary = []
    # 1行ずつ読み込む
    for line in fh:
	wordary = tokenize(line,wordary)
 
    # ファイルを閉じる
    fh.close()

    words[filename] = wordary

# データを作成
def makeData(category):


    return words