#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers
from text_cnn import TextCNN
import csv

# 命令行参数
# ==================================================

# 数据参数
tf.flags.DEFINE_string("test_file", "./data/sms_test3.txt", "用于评估的测试文件地址")
tf.flags.DEFINE_string("stopwords_file", "./data/stopwords.txt", "停用词文件地址")

#词向量模型参数
tf.flags.DEFINE_string("word2vec_model_file", "wiki/wiki.model", "词向量模型文件路径")

# 评估参数
tf.flags.DEFINE_integer("batch_size", 1000, "批处理大小 (默认: 1000)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1555752798/checkpoints", "训练模型时保存的检查点目录")
tf.flags.DEFINE_boolean("eval_train", True, "在所有训练数据上进行评估")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\命令行参数:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 验证
# ==================================================

# 验证检查点文件
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("检查点文件不存在!")
    exit(0)
print("使用检查点文件: {}".format(checkpoint_file))

# 验证word2vec模型文件
if not os.path.exists(FLAGS.word2vec_model_file):
    print("Word2vec模型文件\'{}\' 不存在!".format(FLAGS.word2vec_model_file))
print("使用word2vec模型文件: {}".format(FLAGS.word2vec_model_file))

# 验证训练参数文件
training_params_file = os.path.join(FLAGS.checkpoint_dir, "..", "training_params.pickle")
if not os.path.exists(training_params_file):
    print("训练参数文件 \'{}\' 缺失!".format(training_params_file))
print("使用训练参数文件 : {}".format(training_params_file))

# 加载参数
params = data_helpers.loadDict(training_params_file)
num_labels = int(params['num_labels'])
max_sentence_len = int(params['max_sentence_len'])

# 加载数据
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data(FLAGS.test_file)
else:
    x_raw = ["这是什么", "everything is off."]
    y_test = [1, 0]

#中文分词
sentences = data_helpers.cut_sentences(x_raw)

#停用词过滤
#sentences = data_helpers.filter_stopword(sentences, FLAGS.stopwords_file)

# 获取测试数据集的词向量
x_test = np.array(word2vec_helpers.get_vectors(sentences, FLAGS.word2vec_model_file, max_sentence_len))
y_test = np.array([0 if label=='0' else 1 for label in y_test])
print("x_test.shape = {}".format(x_test.shape))


# 评估
# ==================================================
print("\n评估...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            #print(batch_predictions)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# 打印精确度
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("测试短信总数: {}".format(len(y_test)))
    print("精确度: {:g}".format(correct_predictions/float(len(y_test))))

# 保存评估结果到CSV文件
predictions_human_readable = np.column_stack((np.array([text.encode('utf-8') for text in x_raw]), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("保存评估结果到 {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
