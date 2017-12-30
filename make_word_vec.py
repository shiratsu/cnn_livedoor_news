# -*- coding: utf-8 -*-

from gensim.models import word2vec
import MeCab
import config
import os

class ActionWordVec(object):
    def __init__(self):
        self.model = word2vec.Word2Vec.load('livedoor_corpus.model')


    def getVec(self,word):
        return self.model.wv[word]


def makeVectorData():
    tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/ipadic")

    zero_empty = list = [0 for i in range(100)]

    obj = ActionWordVec()

    all_vec={}

    for dicname in config.DOC_CLASS_DEF:

        filelist = os.listdir('text/' + dicname)

        for filename in filelist:
            if filename == '.DS_Store':
                continue
            fi = open('text/' + dicname + '/' + filename, 'r', encoding='utf-8')

            i = 0
            print(filename)

            ary_vec = []

            line = fi.readline()
            while line:

                if i != 0 and i != 1:

                    node = tagger.parseToNode(line).next
                    while node:

                        vec = obj.getVec(node.surface)

                        if (node.feature.split(",")[0] == "名詞" or node.feature.split(",")[0] == "動詞") \
                                and vec not in ary_vec:
                            ary_vec.append(vec)
                        node = node.next

                line = fi.readline()
                i += 1

            length = len(ary_vec)

            # 500の長さに設定
            if length < 500:

                while length < 500:
                    ary_vec.append(zero_empty)
                    length = len(ary_vec)
            elif length > 500:
                ary_vec = ary_vec[:500]

            all_vec[filename] = ary_vec
    return all_vec