# --*-- coding: utf-8 --*--
import numpy as np
import tensorflow as tf

from .tools.entities import TrainOptions
from .tools.model_snippets import essential_dense_layer, tf_pad_sequence
from .tools.cus_logger import get_logger
from .tools.dictionary import CommonDictionary
from .tools.simple_data_set import BatchDataSet
from .tools.model_snippets import check_early_stop, cal_accuracy, filter_grads_vals


class GeneralTextCNN(object):
    def __init__(self, dictionary: CommonDictionary, train_options: TrainOptions, global_step,
                 learning_rate=0.001, embedding_size=None, filter_sizes=None, num_filters=None):
        """
        :param dictionary:
        :param train_options:
        :param learning_rate:
        :param embedding_size:
        :param filter_sizes:
        :param num_filters:
        """
        self.logger = get_logger(__name__)
        self.vocab = dictionary.w2index
        self.labels = dictionary.label2index

        self.vocab_size = dictionary.dictionary_size
        self.label_size = dictionary.label_size
        self.default_label = "__label__NEUTRAL"
        self.seq_len = dictionary.max_seq_len

        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.UNK = dictionary.UNK
        self.PAD = dictionary.PAD
        self.EOS = dictionary.EOS
        self.UNK_ID = self.vocab[self.UNK]

        self.train_options = train_options

        self.train_summary_op = None
        self.valid_summary_op = None

        self.global_step = global_step
        self.learning_rate = learning_rate

        # two summary tensor. For valid set we need iterate many times, thus a placeholder can help
        self.temp_loss = tf.placeholder(dtype=tf.float32, shape=[], name="temp_loss")
        self.temp_acc = tf.placeholder(dtype=tf.float32, shape=[], name="temp_acc")

        self._build_input()
        self._build_model()
        self._build_optimizer()
        self._build_summary()
        self._customized_operations()
        self.saver = tf.train.Saver(max_to_keep=1)

    def _build_input(self):
        # mainly for inference
        self.logger.info("build_input")
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_keep_prob')
        self.input_x = tf.placeholder(dtype=tf.string, shape=[None, None], name='sentence')
        self.input_y = tf.placeholder(dtype=tf.string, shape=[None, ], name="label")
        self._build_hash_table()
        # input indexing
        input_x = tf_pad_sequence(self.input_x, self.seq_len, self.PAD)
        with tf.variable_scope('input'), tf.device("/cpu:0"):
            self.x_idx = self.symbol2index.lookup(input_x, name="x_idx")
            self.y_idx = self.label2index.lookup(self.input_y, name="y_idx")

    def _build_hash_table(self):
        """
        only used when predict by sentence.
        often times when training, we feed x_idx and y_idx directly
        :return:
        """
        self.logger.info("build_hash_table")
        self.symbol2index = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=self.UNK_ID,
            shared_name="cnn_symbol_table_share",
            name="cnn_symbol_table")

        self.label2index = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=self.label_size,
            shared_name="cnn_label_table_share",
            name="cnn_label_table")

        self.index2label = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value=self.default_label,
            shared_name="cnn_out_table_share",
            name="cnn_out_table")

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
            self.init_table_ops = [insert_vocab_op, insert_labels_op, insert_reverse_labels_op]

    def _build_model(self):
        self.logger.info("build_model")
        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            emb_matrix = tf.get_variable(
                name="emb_matrix",
                initializer=tf.random_uniform_initializer(),
                shape=[self.vocab_size, self.embedding_size],
                dtype=tf.float32)
            x_embedding = tf.nn.embedding_lookup(params=emb_matrix, ids=self.x_idx)  # (batch, vocab_size, embedding)
            x_embedding_expand = tf.expand_dims(x_embedding, axis=-1)    # (batch, vocab_size, embedding, channels)

        # convolution + max-pooling
        conv_pool_outs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("cnn_max_pool_%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                conv = tf.nn.conv2d(
                    input=x_embedding_expand, filter=W,
                    strides=[1, 1, 1, 1], padding="VALID", name="conv")

                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h_per_pool = tf.nn.max_pool(
                    value=h, ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1], padding="VALID", name="max_pool")     # NHWC
                conv_pool_outs.append(h_per_pool)

        # combine all features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(conv_pool_outs, axis=3)  # (batch, 1, 1, num_filters_total)
        self.h_pool_flat = tf.reshape(self.h_pool, shape=[-1, num_filters_total])

        # dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        with tf.variable_scope("output"):
            self.logits = essential_dense_layer(self.h_drop, self.label_size, name='logits')
            self.soft_logits = tf.nn.softmax(self.logits, axis=-1, name="soft_logits")
            self.predictions = tf.argmax(self.soft_logits, axis=1, name="predictions")
            self.label_out = self.index2label.lookup(self.predictions, name="label_out")
            self.topk_scores, self.topk_indices = tf.nn.top_k(self.soft_logits, k=3, name="topk_output")
            self.topk_labels = self.index2label.lookup(tf.cast(self.topk_indices, tf.int64), name="topk_label")

    def _build_optimizer(self):
        self.logger.info("build optimizer")
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_idx, logits=self.logits)
            self.loss = tf.reduce_mean(losses, name='loss')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step, name='train_op')

        grads_and_vars_no_embedding = filter_grads_vals(self.grads_and_vars, lambda v: v.name.find('embedding') < 0)
        self.train_op_no_embedding = optimizer.apply_gradients(grads_and_vars_no_embedding,
                                                               global_step=self.global_step, name='train_op_no_embedding')

    def _build_summary(self):
        self.logger.info("build summary")
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                v_name = v.name.replace(":", "_")
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v_name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v_name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)
        self.loss_summary = tf.summary.scalar("loss", self.temp_loss)
        self.acc_summary = tf.summary.scalar("accuracy", self.temp_acc)

        self.train_summary_op = tf.summary.merge([self.loss_summary, self.acc_summary, self.grad_summaries_merged])
        self.valid_summary_op = tf.summary.merge([self.loss_summary, self.acc_summary])

    def _customized_operations(self):
        # this model may be trained for many times, but each time, the embedding matrix should
        # not be re-trained.
        cur_name = tf.get_variable_scope().name
        self.every_train_reset_ops = [v for v in tf.trainable_variables() if (v.name.find(cur_name) >= 0 and
                                      v.name.find('embedding/emb_matrix') < 0)]
        self.logger.info('every_train_reset_ops={}'.format(','.join([v.name for v in self.every_train_reset_ops])))

    def initialize(self, sess: tf.Session):
        """
        must be called only once!
        :param sess:
        :return:
        """
        self.logger.info("initialize")
        sess.run(self.init_table_ops)

    def evaluate(self, sess: tf.Session, valid_data: BatchDataSet, epoch, global_step, valid_summary_writer=None):
        """
        evaluate performance on valid data set
        :return: average accuracy and average loss
        """
        valid_data.init_iterator()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        loss_list = []
        acc_list = []
        sc_pre_cnt_total = 0
        sc_pre_right_total = 0
        sc_cnt_total = 0
        while True:
            try:
                x_idx, y_idx = valid_data.get_next()
                feed_dict = {
                    self.x_idx: x_idx, self.y_idx: y_idx
                }
                loss, predictions, = sess.run([self.loss, self.predictions],
                    feed_dict, options=run_options)
                loss_list.append(loss)
                accuracy, sc_pre_cnt, sc_pre_right, sc_cnt = cal_accuracy(predictions, y_idx)
                acc_list.append(accuracy)
                sc_pre_cnt_total += sc_pre_cnt
                sc_pre_right_total += sc_pre_right
                sc_cnt_total += sc_cnt
            except IndexError:
                self.logger.info("done read valid data set!")
                break

        avg_loss = np.mean(loss_list)
        avg_acc = np.mean(acc_list)
        self.logger.info(
            "[Valid] epoch {}, step {}, val_loss {:g}, val_acc {:g}. sc_pre_cnt {}, sc_pre_rate {:g}, sc_recall {:g}".
            format(epoch, global_step, avg_loss, avg_acc, sc_pre_cnt_total,
                   0 if sc_pre_cnt_total == 0 else sc_pre_right_total / sc_pre_cnt_total,
                   0 if sc_cnt_total == 0 else sc_pre_right_total / sc_cnt_total)
        )

        if valid_summary_writer:
            feed_dict = {
                self.temp_loss: avg_loss, self.temp_acc: avg_acc
            }
            summaries = sess.run(self.valid_summary_op, feed_dict)
            valid_summary_writer.add_summary(summaries, global_step)

        return avg_acc, avg_loss

    def save(self, sess: tf.Session, saver: tf.train.Saver, ckpt_dir, global_step):
        self.logger.info("saving model ...")
        saver.save(sess, "{}/model.ckpt".format(ckpt_dir), global_step=global_step)

    def train(self, sess: tf.Session, train_X, train_y, valid_X, valid_y,
              train_summary_writer: tf.summary.FileWriter=None, valid_summary_writer:tf.summary.FileWriter=None,
              saver: tf.train.Saver=None, ckpt_dir=None, no_embedding=False):
        self.logger.info("begin to train...")
        train_data = BatchDataSet(train_X, train_y, self.train_options.batch_size, self.train_options.over_sample)
        valid_data = BatchDataSet(valid_X, valid_y, self.train_options.batch_size, self.train_options.over_sample)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        sess.run(tf.variables_initializer(self.every_train_reset_ops))
        early_stop_cnt = 0
        valid_acc = [1e-18]
        valid_losses = [1e18]
        i_epoch = 0
        while i_epoch < self.train_options.max_epoch:
            if check_early_stop(early_stop_cnt, self.train_options.patient):
                self.logger.info("early stop!!")
                break
            try:
                x_idx, y_idx = train_data.get_next()
                feed_dict = {self.x_idx: x_idx,
                             self.y_idx: y_idx,
                             self.dropout_keep_prob: self.train_options.dropout_keep_prob}
                if no_embedding:
                    _, global_step, loss, predictions = sess.run(
                        [self.train_op_no_embedding, self.global_step, self.loss, self.predictions],
                        feed_dict, options=run_options)
                else:
                    _, global_step, loss, predictions = sess.run(
                        [self.train_op, self.global_step, self.loss, self.predictions],
                        feed_dict, options=run_options)
                accuracy, sc_pre_cnt, sc_pre_right, sc_cnt = cal_accuracy(predictions, y_idx)

                feed_dict[self.temp_loss] = loss
                feed_dict[self.temp_acc] = accuracy

                summaries = sess.run(self.train_summary_op, feed_dict)
                self.logger.info(
                    "[Train] epoch {}, step {}, nb_batch {}, loss {:g}, acc {:g}. sc_pre_cnt {}, sc_pre_rate {:g}, sc_recall {:g}".format(
                        i_epoch, global_step, len(y_idx), loss, accuracy, sc_pre_cnt,
                        0 if sc_pre_cnt == 0 else sc_pre_right / sc_pre_cnt,
                        0 if sc_cnt == 0 else sc_pre_right / sc_cnt,
                    ))
                if train_summary_writer:
                    train_summary_writer.add_summary(summaries, global_step)

                if global_step % self.train_options.check_steps == 0:
                    val_accuracy, val_loss = self.evaluate(sess, valid_data, epoch=i_epoch, global_step=global_step,
                                                           valid_summary_writer=valid_summary_writer)
                    early_stop_cnt += 1
                    if val_loss < valid_losses[-1]:  # save best performance
                        valid_acc.append(val_accuracy)
                        valid_losses.append(val_loss)
                        early_stop_cnt = 0
                        if saver and ckpt_dir:
                            self.save(sess, saver, ckpt_dir, global_step)
                            self.logger.info("model improving and saved !")

            except IndexError:
                self.logger.info('done reading train data.')
                train_data.init_iterator()
                i_epoch += 1

    def predict(self, sess: tf.Session, X):
        batch_size = 5000 # 防止CPU、GPU内部不够用，5000条预测一次
        total_num = len(X)
        num_of_batch = (total_num // batch_size) + (0 if total_num % batch_size == 0 else 1)
        list_predictions = []
        list_pred_scores = []
        for i in range(num_of_batch):
            if i == num_of_batch - 1:
                tmpX = X[i*batch_size:]
            else:
                tmpX = X[i*batch_size: i*batch_size+batch_size]
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            feed_dict = {
                self.x_idx: tmpX
            }
            predictions, prediction_scores = sess.run([self.predictions, self.soft_logits],
                                                      feed_dict, options=run_options)
            list_predictions.append(predictions)
            list_pred_scores.append(prediction_scores)
        return np.concatenate(list_predictions), np.concatenate(list_pred_scores)





