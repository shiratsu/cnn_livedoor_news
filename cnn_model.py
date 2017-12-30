# -*- coding: utf-8 -*-

import tensorflow as tf

# 重みを初期化するメソッド
# truncated_normal（切断正規分布）とは正規分布の左右を切り取ったもの
# 重みが0にならないようにしている
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# バイアスを初期化するメソッド
# 0ではなくわずかに陽性=0.1で初期化する
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# xとWを畳み込むメソッド
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 最大プーリング用関数
# プーリングサイズは2x2
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

 # Tensorflowの畳み込み処理
 #    conv = tf.nn.conv2d(
 #        x_expanded,
 #        W_f,
 #        strides=[1, 1, 1, 1],
 #        padding="VALID",
 #        name="conv"
 #    )
 #    # 活性化関数にはReLU関数を利用
 #    h = tf.nn.relu(tf.nn.bias_add(conv, b_f), name="relu")


# # フィルタサイズ：3単語、4単語、5単語の３種類のフィルタ
# filter_sizes = [3, 4, 5]
# # 各フィルタ処理結果をMax-poolingした値をアペンドしていく
# pooled_outputs = []
# for i, filter_size in enumerate(filter_sizes):
#     # フィルタのサイズ（単語数, エンベディングサイズ、チャネル数、フィルタ数）
#     filter_shape = [filter_size, EMBEDDING_SIZE, 1, FILTER_NUM]
#     # フィルタの重み、バイアス
#     W_f = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
#     b_f = tf.Variable(tf.constant(0.1, shape=[FILTER_NUM]), name="b")
#     # Tensorflowの畳み込み処理
#     conv = tf.nn.conv2d(
#         x_expanded,
#         W_f,
#         strides=[1, 1, 1, 1],
#         padding="VALID",
#         name="conv"
#     )
#     # 活性化関数にはReLU関数を利用
#     h = tf.nn.relu(tf.nn.bias_add(conv, b_f), name="relu")
#     # プーリング層 Max Pooling
#     pooled = tf.nn.max_pool(
#         h,
#         ksize=[1, DOCUMENT_LENGTH - filter_size + 1, 1, 1],
#         strides=[1, 1, 1, 1],
#         padding="VALID",
#         name="pool"
#     )
#     pooled_outputs.append(pooled)
#
# # プーリング層の結果をつなげる
# filter_num_total = FILTER_NUM * len(filter_sizes)
# h_pool = tf.concat(3, pooled_outputs)
# h_pool_flat = tf.reshape(h_pool, [-1, filter_num_total])