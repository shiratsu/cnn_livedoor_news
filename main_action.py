# -*- coding: utf-8 -*-

import tensorflow as tf
import cnn_model
import make_data
import config
import make_word_vec
import numpy as np
import random

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

# トレーニング及びテスト
# コスト関数（交差エントロピー）
losses = tf.nn.softmax_cross_entropy_with_logits(scores, y_)
loss = tf.reduce_mean(losses)

# 正答率
correct_predictions = tf.equal(predictions, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

# Adamオプティマイザーによるパラメータの最適化
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-4)
grads_and_vars = optimizer.compute_gradients(loss)
train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


## 実際のトレーニングの処理を作成
all_vec = make_word_vec.makeVectorData()

ary_dokujo = make_word_vec.convertInputOutput(all_vec,'dokujo-tsushin')
ary_itlifehack = make_word_vec.convertInputOutput(all_vec,'it-life-hack')
ary_kadenchannel = make_word_vec.convertInputOutput(all_vec,'kaden-channel')
ary_livedoorhomme = make_word_vec.convertInputOutput(all_vec,'livedoor-homme')
ary_movieenter = make_word_vec.convertInputOutput(all_vec,'movie-enter')
ary_peachy = make_word_vec.convertInputOutput(all_vec,'peachy')
ary_smax = make_word_vec.convertInputOutput(all_vec,'smax')
ary_sportswatch = make_word_vec.convertInputOutput(all_vec,'sports-watch')
ary_topicnews = make_word_vec.convertInputOutput(all_vec,'topic-news')

ary_train = np.r_[ary_dokujo[0]
,ary_itlifehack[0]
,ary_kadenchannel[0]
,ary_livedoorhomme[0]
,ary_movieenter[0]
,ary_peachy[0]
,ary_smax[0]
,ary_sportswatch[0]
,ary_topicnews[0]
]

ary_test = np.r_[ary_dokujo[1]
,ary_itlifehack[1]
,ary_kadenchannel[1]
,ary_livedoorhomme[1]
,ary_movieenter[1]
,ary_peachy[1]
,ary_smax[1]
,ary_sportswatch[1]
,ary_topicnews[1]
]

# 各テンソルのイニシャライズ
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 14000回のイテレーション
for i in range(14000):
    # ミニバッチ（100件ランダムで取得）
    # training_xyには、modelsで定義した各文書行列及び正解ラベル（カテゴリ）が入っている
    samples = random.sample(ary_train, 100)
    batch_xs = [s[0] for s in samples]
    batch_ys = [s[1] for s in samples]
    # 確率的勾配降下法を使い最適なパラメータを求める
    # dropout_keep_probは0.5を指定
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, dropout_keep_prob: 0.5})
    if i % 100 == 0:
        # 100件毎に正答率を表示
        a = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, dropout_keep_prob: 0.5})
        print("TRAINING(%d): %.0f%%" % (i, (a * 100.0)))

# テストでーた
test_samples = random.sample(ary_test, 100)
test_x = test_samples[0][0]
test_y = test_samples[0][1]

# テストデータの正答率
a = sess.run(accuracy, feed_dict={x: test_x, y_: test_y, dropout_keep_prob: 1.0})
print("TEST DATA ACCURACY: %.0f%%" % (a * 100.0))