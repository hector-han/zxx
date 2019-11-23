# -*- coding:utf-8 -*-
"""
支持：
1. bi-lstm
2. bi-gru
3. bi-lstm + attention
"""
from __future__ import print_function
import os
import time
import logging
import datetime
import numpy as np
import tensorflow as tf
from .tools.cus_logger import get_logger
from .tools.model_snippets import padding


def cal_accuracy(predictions, label, code=2):
    correct_predictions = np.equal(predictions, label)
    accuracy = np.mean(correct_predictions.astype(float))

    tmp = np.equal(predictions, np.zeros_like(predictions) + code)
    tmp_ints = tmp.astype(int)
    code_pre_cnt = np.sum(tmp_ints)

    tmp1 = np.equal(label, np.zeros_like(label) + code)
    code_pre_right = np.sum(np.logical_and(tmp, tmp1).astype(int))

    return accuracy, code_pre_cnt, code_pre_right


class RNNModel(object):
    def __init__(self, dictionary, args, graph_from_meta=None):
        self.logger = get_logger(__name__)
        self.vocab = dictionary.w2index
        self.labels = dictionary.label2index

        self.enc_layer = 1
        self.use_lstm = True
        self.dropout_keep_prob = 1.0
        self.optimize = "adam"
        self.momentum = 0.0

        # 解析词典
        self.vocab_size = dictionary.dictionary_size
        self.label_size = dictionary.label_size
        self.seq_len = dictionary.max_seq_len

        # 解析 args 中的参数
        self.rnn_size = args.embedding_dim
        self.num_hidden = args.num_hidden
        self.active = args.active
        self.learning_rate = args.learning_rate
        self.decay_rate = args.decay_rate

        self.step_time = 0.0  # 记录每个 step 的时间

        ##################
        self.PAD = dictionary.PAD
        self.PADID = dictionary.w2index[self.PAD]

        self.train_summary_op = None
        self.valid_summary_op = None
        self.train_summary_writer = None
        self.valid_summary_writer = None

        if graph_from_meta is None:
            self.epoch = tf.Variable(0, dtype=tf.int32, name="epoch")
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            self.temp_loss = tf.Variable(0.0, name="temp_loss", dtype=tf.float32)
            self.temp_acc = tf.Variable(0.0, name="temp_acc", dtype=tf.float32)

            self._build_placeholder()
            self._build_hash_table()
            self._build_model()
            self._optimize()
            self._build_summary()
            self.saver = tf.train.Saver(max_to_keep=1)
        else:
            self.saver = tf.train.import_meta_graph(graph_from_meta)

    def _build_placeholder(self):
        self.logger.info("[LSTM] build_placeholder")
        self.input_x = tf.placeholder(dtype=tf.string, shape=[None, None], name="sentence")
        self.input_y = tf.placeholder(dtype=tf.string, shape=[None,], name="label")
        self.x_len = tf.placeholder(tf.int64, [None, ], name="length")

    def _build_hash_table(self):
        self.logger.info("[LSTM] build_hash_table")
        self.symbol2index = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=self.PADID,
            shared_name="symbol_table_share",
            name="symbol_table")

        self.label2index = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=0,
            shared_name="label_table_share",
            name="label_table")

        self.index2label = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value="",
            shared_name="out_table_share",
            name="out_table")

    def _build_model(self):
        with tf.variable_scope("input"), tf.device("/cpu:0"):
            self.x_idx = self.symbol2index.lookup(self.input_x, name="x_idx")
            self.y_idx = self.label2index.lookup(self.input_y, name="y_idx")

        with tf.variable_scope("embedding", initializer=tf.orthogonal_initializer()), tf.device("/cpu:0"):
            with tf.device('/cpu:0'):
                emb_matrix = tf.get_variable('w', [self.vocab_size, self.rnn_size])
                x_embedded = tf.nn.embedding_lookup(params=emb_matrix, ids=self.x_idx)  # (B, T, run_size)

        with tf.variable_scope("rnn", initializer=tf.orthogonal_initializer()):
            activation_func = tf.nn.softsign if self.active == "softsign" else tf.nn.tanh
            rnn_cell_func = tf.nn.rnn_cell.LSTMCell if self.use_lstm else tf.nn.rnn_cell.GRUCell

            # forward cell
            cells = []
            for _ in range(self.enc_layer):
                rnn_cell = rnn_cell_func(self.num_hidden, forget_bias=1.0, activation=activation_func)
                dropout = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, input_keep_prob=self.dropout_keep_prob)
                cells.append(dropout)
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells)

            # backward cell
            cells = []
            for _ in range(self.enc_layer):
                rnn_cell = rnn_cell_func(self.num_hidden, forget_bias=1.0, activation=activation_func)
                dropout = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, input_keep_prob=self.dropout_keep_prob)
                cells.append(dropout)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells)

            # bi-rnn
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=x_embedded,
                sequence_length=self.x_len, dtype=tf.float32)

        final_output = tf.concat([states[0][0].h, states[1][0].h], axis=1)

        with tf.variable_scope("output"):
            W = tf.get_variable(
                name="W", shape=[2 * self.num_hidden, self.label_size],
                initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.nn.l2_loss)
            b = tf.get_variable(
                name="b", initializer=tf.zeros([self.label_size]), regularizer=tf.nn.l2_loss)
            self.logits = tf.nn.xw_plus_b(x=final_output, weights=W, biases=b, name="logits")

        with tf.name_scope("inference"):    # 得到 top3 的候选
            self.prob_dist = tf.nn.softmax(self.logits)
            self.topk_scores, self.topk_indices = tf.nn.top_k(self.prob_dist, k=3, name="output")
            self.topk_labels = self.index2label.lookup(tf.cast(self.topk_indices, tf.int64), name="label_out")

    def _optimize(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_idx)
        self.loss = tf.reduce_mean(losses)

        # 准确率
        self.predictions = tf.argmax(self.prob_dist, 1)
        correct = tf.equal(self.predictions, self.y_idx)
        self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        # 最优化
        if self.optimize == "adam":
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimize == "rms":
            opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay_rate, momentum=self.momentum)
        else:
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        self.train_op = opt.minimize(self.loss, global_step=self.global_step)

    def _build_summary(self):
        self.loss_summary = tf.summary.scalar("loss", self.temp_loss)
        self.acc_summary = tf.summary.scalar("accuracy", self.temp_acc)
        self.epoch_summary = tf.summary.scalar("epoch", self.epoch)

    def initialize(self, sess):
        logging.info("variables initializing...")
        sess.run(tf.global_variables_initializer())

        logging.info("vocab table inserting...")
        with tf.device("/cpu:0"):
            insert_vocab_op = self.symbol2index.insert(
                keys=tf.constant(list(self.vocab.keys()), dtype=tf.string),
                values=tf.constant(list(self.vocab.values()), dtype=tf.int64))
            insert_labels_op = self.label2index.insert(
                keys=tf.constant(list(self.labels.keys()), dtype=tf.string),
                values=tf.constant(list(self.labels.values()), dtype=tf.int64))
            insert_reverse_labels_op = self.index2label.insert(
                keys=tf.constant(list(self.labels.values()), dtype=tf.int64),
                values=tf.constant(list(self.labels.keys()), dtype=tf.string))
            sess.run([insert_vocab_op, insert_labels_op, insert_reverse_labels_op])

    def step(self, sess, batch_data, epoch):
        tic = time.time()
        # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        feed_dict = {
            "sentence:0": batch_data["sentence"],
            "label:0": batch_data["label"],
            "length:0": batch_data["length"]
        }
        sess.run(tf.assign(self.epoch, epoch))
        batch_num = len(batch_data["sentence"])
        _, epoch, step, summaries, loss, accuracy, predictions, y_idx = sess.run(
            [
                self.train_op, self.epoch, self.global_step, self.train_summary_op,
                self.loss, self.acc, self.predictions, self.y_idx
            ],
            feed_dict=feed_dict)
        self.step_time = time.time() - tic
        assign_loss_op = tf.assign(self.temp_loss, loss)
        assign_acc_op = tf.assign(self.temp_acc, accuracy)
        sess.run([assign_loss_op, assign_acc_op])

        accuracy, sc_pre_cnt, sc_pre_right = cal_accuracy(predictions, y_idx)
        self.logger.info("[Train]: epoch {}, step {}, nb_batch {}, loss {:g}, acc {:g}|| sc_pre_cnt {}, sc_pre_rate {:g}".format(
            epoch, step, batch_num, loss, accuracy, sc_pre_cnt, 0 if sc_pre_cnt == 0 else sc_pre_right / sc_pre_cnt))
        if self.train_summary_writer:
            self.train_summary_writer.add_summary(summaries, step)

    def evaluate(self, sess, valid_data_loader, epoch, global_step):
        loss_list = []
        acc_list = []
        sc_pre_cnt_total = 0
        sc_pre_right_total = 0
        valid_data_loader.one_shot_init()
        while True:
            try:
                batch_data = valid_data_loader.one_shot_batch(sess)
                feed_dict = {
                    "sentence:0": batch_data["sentence"],
                    "label:0": batch_data["label"],
                    "length:0": batch_data["length"]
                }
                loss, accuracy, predictions, y_idx = sess.run([self.loss, self.acc, self.predictions, self.y_idx], feed_dict)
                loss_list.append(loss)
                acc_list.append(accuracy)

                accuracy, sc_pre_cnt, sc_pre_right = cal_accuracy(predictions, y_idx)
                sc_pre_cnt_total += sc_pre_cnt
                sc_pre_right_total += sc_pre_right
            except:
                self.logger.info('done read one shot data')
                break

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)

        time_str = datetime.datetime.now().isoformat()
        self.logger.info("[Valid] {}: epoch {}, step {}, val_loss {:g}, val_acc {:g} || sc_pre_cnt {}, sc_pre_rate {:g}".format(
            time_str, epoch, global_step, avg_loss, avg_acc, sc_pre_cnt_total,
            0 if sc_pre_cnt_total == 0 else sc_pre_right_total / sc_pre_cnt_total))

        if self.valid_summary_writer:
            assign_loss_op = tf.assign(self.temp_loss, avg_loss)
            assign_acc_op = tf.assign(self.temp_acc, avg_acc)
            sess.run([assign_loss_op, assign_acc_op])
            summaries = sess.run(self.valid_summary_op)
            self.valid_summary_writer.add_summary(summaries, global_step)

        return avg_acc, avg_loss

    def summary(self, graph, log_dir):
        self.logger.info("[RNNModel] summary")
        # Train Summaries
        train_summary_dir = os.path.join(log_dir, "train")
        self.train_summary_op = tf.summary.merge(
            [self.epoch_summary, self.loss_summary, self.acc_summary])
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph)

        # Dev summaries
        valid_summary_dir = os.path.join(log_dir, "valid")
        self.valid_summary_op = tf.summary.merge([self.loss_summary, self.acc_summary])
        self.valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, graph)

    def save(self, sess, ckpt_dir, global_step):
        self.logger.info("[RNNModel] saving...")
        self.saver.save(sess, "%s/model.ckpt" % ckpt_dir, global_step=global_step)

    def load(self, sess, ckpt_dir):
        self.logger.info("[RNNModel] loading...")
        self.saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    def inference(self, sess, sentence_batch, seq_len, length_batch=None):
        """
        :param sess:
        :param sentence_batch: 分词后的文本
        :param seq_len:
        :param length_batch
        :return:
        """
        graph = sess.graph
        if isinstance(sentence_batch, list):
            sentence_batch = np.array(sentence_batch)
        if len(sentence_batch.shape) != 2:
            sentence_batch = np.expand_dims(sentence_batch, axis=0)

        sentences = []
        labels = []
        lengthes = []
        for i, s in enumerate(sentence_batch):
            if length_batch is not None:
                lengthes.append(length_batch[i])
            else:
                lengthes.append(len(s))
            sentences.append(padding(
                sent=s, max_len=seq_len,
                eos=self.PAD, pad=self.PAD))
            labels.append("")

        feed_dict = {
            "sentence:0": sentences,
            "label:0": labels,
            "length:0": lengthes
        }

        x_idx = graph.get_tensor_by_name("input/x_idx:0")

        topk_labels = graph.get_tensor_by_name("inference/label_out:0")
        topk_scores = graph.get_tensor_by_name("inference/output:0")
        topk_indices = graph.get_tensor_by_name("inference/output:1")

        pred_x_idx, topk_labels, topk_scores, topk_indices = sess.run(
            [x_idx, topk_labels, topk_scores, topk_indices], feed_dict=feed_dict)

        return pred_x_idx, topk_indices, topk_labels, topk_scores


