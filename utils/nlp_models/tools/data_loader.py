# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import json
from .model_snippets import padding
from .cus_logger import get_logger
from .constants import VocabParams

logger = get_logger(__name__)


class DataLoader(object):
    def __init__(self, filename, max_seq_len, batch_size=None, mode="train"):
        self.filename = filename
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.batch_size = batch_size

        self.label = 'label'
        self.content = 'content'
        self.epoch = 0
        # 转换函数，比如可以实现对list of words 的 ngram
        with tf.device("/cpu:0"):
            self.base_dataset = tf.data.TextLineDataset(self.filename)
            self.one_shot_iterator = self.base_dataset.batch(self.batch_size).make_one_shot_iterator()
            self.one_shot_element = self.one_shot_iterator.get_next()
        self._define_loop_iterator()

    def _define_loop_iterator(self):
        """
        可以循环迭代的iterator
        :return:
        """
        logger.info("Init _define_loop_iterator...")
        with tf.device("/cpu:0"):
            shuffle_dataset = self.base_dataset.shuffle(2 * self.batch_size)
            batch_dataset = shuffle_dataset.batch(self.batch_size)
            self.loop_iterator = batch_dataset.make_one_shot_iterator()
            self.next_loop_element = self.loop_iterator.get_next()

    def next_loop_batch(self, sess: tf.Session):
        with tf.device("/cpu:0"):
            run_options = None
            try:
                # list_of_string, with batch_size
                batch_data = sess.run(self.next_loop_element, options=run_options)
            except tf.errors.OutOfRangeError:
                logger.info('Done reading loop batch')
                self.epoch += 1
                self._define_loop_iterator()
                batch_data = sess.run(self.next_loop_element, options=run_options)

        out_batch_data = self._process_batch(batch_data)
        return out_batch_data

    def one_shot_init(self):
        with tf.device("/cpu:0"):
            self.one_shot_iterator = self.base_dataset.batch(self.batch_size).make_one_shot_iterator()
            self.one_shot_element = self.one_shot_iterator.get_next()

    def one_shot_batch(self, sess: tf.Session):
        with tf.device("/cpu:0"):
            batch_data = sess.run(self.one_shot_element)
            out_batch_data = self._process_batch(batch_data)
            return out_batch_data

    def _check(self, dict_data) -> bool:
        if self.label not in dict_data:
            logger.warning(self.label + " not found")
            return False
        if self.content not in dict_data:
            logger.warning(self.content + " not found")
            return False
        return True

    def _process_batch(self, batch_data):
        sentences = []
        labels = []
        sent_lengthes = []
        tweet_ids = []
        for line in batch_data:
            line = line.decode()
            try:
                line_json = json.loads(line)
            except json.JSONDecodeError:
                logger.error(line)
                continue
            if not self._check(line_json):
                continue
            label, text, tid = line_json[self.label], line_json[self.content], line_json['tid']
            sent_lengthes.append(len(text))
            sentences.append(padding(sent=text, max_len=self.max_seq_len, eos=VocabParams.eos, pad=VocabParams.pad))
            # vectors.append(vector)

            label = VocabParams.label_prefix + label
            labels.append(label)
            tweet_ids.append(tid)

        out_batch_data = {
            "sentence": np.array(sentences),
            "label": np.array(labels),
            "length": np.array(sent_lengthes),
            "tid": np.array(tweet_ids),
            # "vector": np.array(vectors)
        }

        return out_batch_data