# -*- coding: utf-8 -*-

import tensorflow as tf
import cnn_model
import make_data
import config

# 定数
# 単語ベクトルのサイズ
EMBEDDING_SIZE = 100
# １フィルター種類に対するフィルターの個数
FILTER_NUM = 128
# 1文書に含まれる単語数（全文書合わせてある）
DOCUMENT_LENGTH = 500


# 変数
# インプット変数（各文書が500 x 100のマトリクス）
x = tf.placeholder(tf.float32, [None, DOCUMENT_LENGTH, EMBEDDING_SIZE], name="x")
# アプトプット変数（文書カテゴリー）
y_ = tf.placeholder(tf.float32, [None, len(config.DOC_CLASS_DEF)], name="y_")
# ドロップアウト変数
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# インプット変数の次元を拡張しておく（channel）
x_expanded = tf.expand_dims(x, -1)

## モデルを作成
### 第一畳み込み層 ####################################################################

# 重みの初期化
W_conv1 = cnn_model.weight_variable([3, EMBEDDING_SIZE, 1, FILTER_NUM])
b_conv1 = cnn_model.bias_variable([FILTER_NUM])
x_text = tf.reshape(x_expanded, [-1,28,28,1])
h_conv1 = tf.nn.relu(cnn_model.conv2d(x_text, W_conv1) + b_conv1)
h_pool1 = cnn_model.max_pool(h_conv1)

### 第二畳み込み層 ####################################################################

# 重みの初期化
# 第一層で出力が32だったので、入力チャンネルは32となる
W_conv2 = cnn_model.weight_variable([4, EMBEDDING_SIZE, FILTER_NUM, 256])
b_conv2 = cnn_model.bias_variable([256])
h_conv2 = tf.nn.relu(cnn_model.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = cnn_model.max_pool(h_conv2)

### 第二畳み込み層 ####################################################################

# 重みの初期化
# 第一層で出力が32だったので、入力チャンネルは32となる
W_conv3 = cnn_model.weight_variable([5, EMBEDDING_SIZE, 256, 512])
b_conv3 = cnn_model.bias_variable([512])
h_conv3 = tf.nn.relu(cnn_model.conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = cnn_model.max_pool(h_conv3)

### 密に接続された層 ##################################################################

# # プーリング層の結果をつなげる
filter_num_total = FILTER_NUM * len(3)
h_pool_flat = tf.reshape(h_pool3, [-1, filter_num_total])

# ドロップアウト（トレーニング時0.5、テスト時1.0）
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# アウトプット層
class_num = len(config.DOC_CLASS_DEF)
W_o = tf.Variable(tf.truncated_normal([filter_num_total, class_num], stddev=0.1), name="W")
b_o = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b")
scores = tf.nn.xw_plus_b(h_drop, W_o, b_o, name="scores")
predictions = tf.argmax(scores, 1, name="predictions")