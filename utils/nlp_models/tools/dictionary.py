# -*- coding:utf-8 -*-
import os
import json

from .constants import VocabParams
from .cus_logger import get_logger
from .model_snippets import map_content_to_index

"""
构建一个文本分类的字典，作用和tensorflow的VocabularyProcessor一样，把 一个字符串 和 一个整数id 互相转换。标签也一样处理
之所以自己重写一个，是想可以自己控制细节。
"""


class CommonDictionary(object):
    def __init__(self, min_count=0, seq_limit=None, max_num_words=None, dict_dir=None, text_indicator=['content']):
        """
        增加支持n-gram.
        :param min_count: 最少出现次数，比这小的忽略
        :param seq_limit: 手动指定句子最多包含的word（n-gram认为1个word）个数，如果大于这个数，这个句子不参与构建字典。
        :param max_num_words: 字典包含的最大word个数，如果超过，丢弃频次低的
        :param dict_dir: 词典路径，保存现有的，或者加载已有的
        :param text_indicator: 按行读取json格式的一行，指明了那个是需要处理的文本
        """
        self.logger = get_logger(__name__)
        self.min_count = min_count
        self.seq_limit = seq_limit
        self.max_num_words = max_num_words
        self.text_indicator = text_indicator

        self.UNK = VocabParams.unk
        self.PAD = VocabParams.pad
        self.EOS = VocabParams.eos
        self.label_prefix = VocabParams.label_prefix

        self.max_seq_len = 0
        self.min_seq_len = 10000

        self.init_words = [self.UNK, self.PAD, self.EOS]
        self.index2w = [e for e in self.init_words]
        self.w2index = map_content_to_index(self.index2w)
        self.w2cnt = dict()  # 统计单词在语料中出现的词频
        self.w2df = dict()  # 保存词典中的关键词的文档频率
        self.index2label = []
        self.label2index = dict()
        self.label2cnt = dict()

        # 保存分好词的语料
        self.load_update_dictionary(dict_dir=dict_dir)

    @property
    def dictionary_size(self):
        return len(self.w2index)

    @property
    def label_size(self):
        return len(self.label2index)

    def read_corpus(self, text_iter=None, transfer_func: callable(list)=None):
        """
        :param text_iter: text_iter每次返回的是一个dict对象，text_indicator指示文本内容对应的key是啥
        :param transfer_func
        :return:
        """
        if text_iter is None:
            self.logger.warning("text_iter is None.")
            return

        self._travese_content(text_iter, transfer_func)
        self.logger.info("Traverse corpus completed! Raw Dictionary length is %d" % len(self.w2index))
        self._check()
        self.logger.info("[Dictionaries] Final Dictionary length is %d" % len(self.w2index))

    def _travese_content(self, text_iter, transfer_func: callable(list)):
        """
        迭代text_iter中的每一行，读取数据，真正的构造字典的函数
        :param text_iter:
        :param transfer callable
        :return:
        """
        for i, (text_dict, label) in enumerate(text_iter):
            # text 是一个dcit对象，
            for indicator in self.text_indicator:
                temp_words = text_dict[indicator]
                if isinstance(temp_words, str):
                    temp_words = temp_words.split(" ")

                # 这儿删掉了一些没用的处理
                words = temp_words
                # 进行转换，主要是
                if transfer_func:
                    words = transfer_func(words)

                if self.seq_limit and len(words) > self.seq_limit:
                    continue

                if len(words) > self.max_seq_len:
                    self.max_seq_len = len(words)
                if len(words) < self.min_seq_len:
                    self.min_seq_len = len(words)

                # 统计 label
                label_str = self.label_prefix + label
                if label_str not in self.label2index:
                    self.label2index[label_str] = len(self.label2index)
                    self.label2cnt[label_str] = 0
                self.label2cnt[label_str] += 1

                # 构建词典，统计词频
                for cw in words:
                    cw = cw.strip()
                    if cw not in self.w2index:
                        self.w2index[cw] = len(self.w2index)
                    if cw not in self.w2cnt:
                        self.w2cnt[cw] = 1
                    else:
                        self.w2cnt[cw] += 1

                # 构建单词的文档频率
                for cw in set(words):
                    if cw not in self.w2df:
                        self.w2df[cw] = 1
                    else:
                        self.w2df[cw] += 1
            self._get_index2w()

    def _get_index2w(self):
        """
        根据w2index获得index2w
        :return:
        """
        self.index2w = (max(self.w2index.values()) + 1) * [self.UNK]
        for w, i in self.w2index.items():
            self.index2w[i] = w

        self.index2label = (max(self.label2index.values()) + 1) * [self.UNK]
        for w, i in self.label2index.items():
            self.index2label[i] = w

        return self.index2w

    def _check(self):
        """
        首先检查是否小于max_num_words , 然后再检查大于等于min_count。最终结果有可能小于max_num_words
        :return:
        """
        if self.max_num_words and len(self.w2cnt) > self.max_num_words:
            self.logger.warning("dict size > max_num_words, remove some")
            new_w2cnt = list(sorted(self.w2cnt.items(), key=lambda x: x[1], reverse=True)[:self.max_num_words])
            new_wdf = dict()
            self.w2cnt = dict()
            for w, c in new_w2cnt:
                self.w2cnt[w] = c
                new_wdf[w] = self.w2df[w]
            self.w2df = new_wdf

        new_index2w = list(self.init_words)
        for w in self.index2w:
            if w not in self.init_words and w in self.w2cnt and self.w2cnt[w] >= self.min_count:
                new_index2w.append(w)
        self.index2w = new_index2w
        self.w2index = map_content_to_index(self.index2w)

    def _make_file_name(self, dict_dir):
        idx2w_fn = os.path.join(dict_dir, "idx2w.dic")
        w2cnt_fn = os.path.join(dict_dir, "w2cnt.json")
        labels_dic = os.path.join(dict_dir, "labels.dic")
        w2df_fn = os.path.join(dict_dir, "w2df.json")
        label_cnt = os.path.join(dict_dir, "label2cnt.json")
        return idx2w_fn, w2cnt_fn, w2df_fn, labels_dic, label_cnt

    def load_update_dictionary(self, dict_dir):
        """
        载入原来的字典，与当前的字典做合并
        如果提供了词典文件，则在原来词典的基础上更新词典
        更新内容：index2w & w2index & w2cnt & w2df_fn
        :param dict_dir: one line for one word or char
        :return:
        """
        if dict_dir is None:
            self.logger.info("not update dictionary from dict_dir (dict_dir is None).")
            return
        self._load_model(dict_dir)

        idx2w_fn, w2cnt_fn, w2df_fn, labels_dic, label_cnt = self._make_file_name(dict_dir=dict_dir)
        if not os.path.exists(idx2w_fn):
            self.logger.info("not update dictionary from {} (is empty).".format(idx2w_fn))
            return

        # 加载/更新词典索引【index2w & w2index】
        dict_loaded = []
        with open(idx2w_fn, 'r', encoding="utf-8") as fp:
            for line in fp:
                line = line.strip('\n').strip('\r').strip('\t')
                if len(line) == 0:
                    self.logger.error("encounter an empty line")
                    continue
                dict_loaded.append(line)
        # 把当前字典的内容，加入从文件load的字典中去
        for ch in self.index2w:
            if ch in dict_loaded:
                continue
            else:
                dict_loaded.append(ch)
        self.index2w = dict_loaded
        self.w2index = map_content_to_index(self.index2w)

        # 加载/更新标签索引【index2label & label2index】
        labels_loaded = []
        with open(labels_dic, 'r', encoding="utf-8") as fp:
            for line in fp:
                line = line.strip('\n').strip('\r').strip('\t')
                if len(line) == 0:
                    continue
                labels_loaded.append(line)
        for i in self.index2label:
            if i in labels_loaded:
                continue
            else:
                labels_loaded.append(i)
        self.index2label = labels_loaded
        self.label2index = map_content_to_index(self.index2label)

        # 加载/更新词频词典【w2cnt】
        with open(w2cnt_fn, 'r', encoding='utf-8') as fp:
            w2cnt = json.loads(fp.read())
        for w, c in w2cnt.items():
            if w not in self.w2cnt:
                self.w2cnt[w] = c
            else:
                self.w2cnt[w] += c

        # 加载/更新文档频率词典【w2df】
        with open(w2df_fn, 'r', encoding='utf-8') as fp:
            w2df = json.loads(fp.read())
        for w, c in w2df.items():
            if w not in self.w2df:
                self.w2df[w] = c
            else:
                self.w2df[w] += c

        with open(label_cnt, 'r', encoding='utf-8') as fp:
            label2cnt = json.loads(fp.read())
        for w, c in label2cnt.items():
            if w not in self.label2cnt:
                self.label2cnt[w] = c
            else:
                self.label2cnt[w] += c

        self._check()

        self.logger.info("Load %s completed! Dictionary length is %d" % (
            idx2w_fn, len(self.w2index)))

        self.logger.info("Load %s completed! Dictionary length is %d" % (
            w2cnt_fn, len(self.w2cnt)))

        self.logger.info("Load %s completed! Dictionary length is %d" % (
            w2df_fn, len(self.w2df)))

    def save_dictionary(self, dict_dir):
        """
        one line for one word or char
        :param dict_dir:
        :return:
        """
        if not os.path.exists(dict_dir):
            os.makedirs(dict_dir)
        self._save_model(dict_dir)
        idx2w_fn, w2cnt_fn, w2df_fn, labels_dic, label_cnt = self._make_file_name(dict_dir=dict_dir)

        # 保存词典
        with open(idx2w_fn, 'w', encoding="utf-8") as fp:
            for w in self.index2w:
                fp.write(w + u"\n")

        with open(w2cnt_fn, 'w', encoding="utf-8") as fp:
            json_str = json.dumps(self.w2cnt)
            fp.write(json_str)

        with open(label_cnt, 'w', encoding="utf-8") as fp:
            json_str = json.dumps(self.label2cnt)
            fp.write(json_str)

        with open(w2df_fn, 'w', encoding="utf-8") as fp:
            json_str = json.dumps(self.w2df)
            fp.write(json_str)

        self.logger.info("[Dictionaries] Save %s completed! Dictionary length is %d" % (
            idx2w_fn, len(self.w2index)))

        self.logger.info("[Dictionaries] Save %s completed! Dictionary length is %d" % (
            w2cnt_fn, len(self.w2cnt)))

        self.logger.info("[Dictionaries] Save %s completed! Dictionary length is %d" % (
            w2df_fn, len(self.w2df)))

        # 保存标签
        with open(labels_dic, 'w', encoding="utf-8") as fp:
            self.index2label.sort()
            for w in self.index2label:
                fp.write(w + u"\n")


    def _save_model(self, dict_dir):
        """
        保存汇总信息
        :param dict_dir:
        :return:
        """
        params = {
            "max_seq_len": self.max_seq_len,
            "min_seq_len": self.min_seq_len,
            "seq_limit": self.seq_limit,
            "min_count": self.min_count,
            "max_num_words": self.max_num_words
        }
        json_str = json.dumps(params)

        if not os.path.exists(dict_dir):
            os.makedirs(dict_dir)

        fn = os.path.join(dict_dir, "vocab_model.json")
        with open(fn, 'w') as fp:
            fp.write(json_str)
        self.logger.info("[Dictionaries] model saved into %s" % fn)

    def _load_model(self, dict_dir):
        """
        加载之前的字典的汇总信息，如果路径不存在，则创建
        :param model_dir: 之前的字典路径
        :return:
        """
        if dict_dir is None:
            return
        if not os.path.exists(dict_dir):
            os.makedirs(dict_dir)
            return
        fn = os.path.join(dict_dir, "vocab_model.json")
        if not os.path.exists(fn):
            self.logger.info("[Dictionaries] not load_model dictionary from model_dir (is empty).")
            return

        with open(fn, 'r') as fp:
            json_str = fp.read()
            self.logger.info("will load dict = {}".format(json_str))
        params = json.loads(json_str)

        self.max_seq_len = params["max_seq_len"]
        self.min_seq_len = params["min_seq_len"]
        self.seq_limit = params["seq_limit"]
        self.min_count = params["min_count"]
        self.max_num_words = params["max_num_words"]
        self.logger.info("[Dictionaries] model loaded from %s" % fn)
