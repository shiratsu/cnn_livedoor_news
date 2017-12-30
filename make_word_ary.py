# -*- coding: utf-8 -*-

import MeCab
import sys
import os
import config

tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/ipadic")


fo = open(sys.argv[1], 'w')
allkeywords = []
for dicname in config.DOC_CLASS_DEF:

    filelist = os.listdir('text/'+dicname)

    for filename in filelist:
        if filename == '.DS_Store':
            continue
        fi = open('text/'+dicname+'/'+filename, 'r', encoding='utf-8')

        i = 0
        print(filename)

        line = fi.readline()
        while line:

            if i != 0 and i != 1:
                # line = line.encode("utf-8")
                # line = line.strip()
                # line = str(line).replace('b', '')
                # print(line)
                # print(i)
                node = tagger.parseToNode(line).next
                while node:
                    if (node.feature.split(",")[0] == "名詞" or node.feature.split(",")[0] == "動詞") \
                            and node.surface not in allkeywords:
                        allkeywords.append(node.surface)
                    node = node.next

            line = fi.readline()
            i+=1

        fi.close()


fo.write(" ".join(allkeywords))  # skip first \s
fo.close()