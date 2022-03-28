import tensorflow as tf
import numpy as np


class TextCNN(object):
    '''
    用于文本分类的卷积神经网络模型
    输入层，卷积层，最大池化层，全连接层
    '''
    
    def __init__(self, text_len, num_classes, embedding_size,
                 conv_sizes, num_convs, l2_reg_lambda=0.0):
        
        #输入，输出，dropout占位符
        self.input_x = tf.placeholder(tf.float32, [None, text_len, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        #跟踪L2正则化损失（可选） 
        l2_loss = tf.constant(0.0)

        #输入层
        # self.embedded_chars = [None(batch_size), sequence_size, embedding_size]
        # self.embedded_chars = [None(batch_size), sequence_size, embedding_size, 1(num_channels)]
        self.embedded_chars  = self.input_x
        self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)
        
        #为每种卷积核尺寸创建一个卷积+池化层
        pooled_outputs = []
        for i,conv_size in enumerate(conv_sizes):
            with tf.name_scope("conv-maxpool-%s" % conv_size):
                #卷积层
                filter_shape = [conv_size, embedding_size, 1, num_convs]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_convs]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expended, W, strides=[1,1,1,1], padding="VALID", name="conv")
              
                #激励层
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
                #池化层
                pooled = tf.nn.max_pool(h, 
                          ksize=[1,text_len - conv_size + 1, 1, 1],
                          strides = [1, 1, 1, 1],
                          padding = "VALID",
                          name = "pool")
                pooled_outputs.append(pooled)
         
        # 连接所有的池化特征
        num_convs_total = num_convs * len(conv_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_convs_total])
                
        #添加dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        #最终（非正规化）分数和预测 
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_convs_total, num_classes],
                                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], name="b"))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
      
        #计算平均交叉熵损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        #准确度
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")