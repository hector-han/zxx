# --*-- coding: utf-8 --*--
import tensorflow as tf
import numpy as np
import json
import os


"""
some tf simple layers and functions
"""

def essential_dense_layer(input, num_output, name=None):
    """
    a dense layer with w[:,0] = 0, b[0]= 0. useful for the last softmax layer
    :param input: input tensor, [batch, in_dim]
    :param num_output: number of output nodes, must > 1
    :param name: name of logits
    :return: logits
    """
    in_dim = input.shape[1]
    w0 = tf.zeros(shape=[in_dim, 1], dtype=tf.float32)
    b0 = tf.zeros(shape=[1, ], dtype=tf.float32)
    w1 = tf.get_variable(name="w1", shape=[in_dim, num_output - 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.constant(0.0, shape=[num_output - 1]))
    w = tf.concat((w0, w1), axis=1)
    b = tf.concat((b0, b1), axis=0)
    logits = tf.nn.xw_plus_b(x=input, weights=w, biases=b, name=name)
    return logits


def tf_pad_sequence(input_x, seq_len, pad):
    """
    在 tensorflow 的 graph 中完成输入的padding工作，
    优势：使用CNN网络时，不需要再外传 seq_len
    注意：这里只会padding <pad> 中的内容，若想加入 eos 符号，仍然需要在graph外面添加
    :param input_x: 输入 tensor，维度为 (batch_size, seq_len)
        如：tf.placeholder(dtype=tf.string, shape=[None, None], name="sentence")
    :param seq_len: int，需要padding的最大长度，如果序列长度超过 seq_len，则会自动截断
    :param pad: 需要padding在后面的符号，注意，类型需要和 input_x 中一致
    :return:
    """
    with tf.variable_scope("padding"), tf.device("/cpu:0"):
        input_shape = tf.shape(input_x)
        min_len = tf.minimum(input_shape[1], seq_len)
        input_slice = tf.slice(input_x, [0, 0], [-1, min_len])
        input_padding = tf.fill([input_shape[0], seq_len-min_len], pad)
        input_x = tf.concat([input_slice, input_padding], axis=1)
        return tf.reshape(input_x, shape=[-1, seq_len])


def check_early_stop(cnt, patient):
    stop_flag = patient and cnt >= patient
    return stop_flag


def cal_accuracy(predictions, label, code=2):
    correct_predictions = np.equal(predictions, label)
    accuracy = np.mean(correct_predictions.astype(float))

    tmp = np.equal(predictions, np.zeros_like(predictions) + code)
    tmp_ints = tmp.astype(int)
    code_pre_cnt = np.sum(tmp_ints)

    tmp1 = np.equal(label, np.zeros_like(label) + code)
    tmp_ints1 = tmp1.astype(int)
    code_true_cnt = np.sum(tmp_ints1)

    code_pre_right = np.sum(np.logical_and(tmp, tmp1).astype(int))

    return accuracy, code_pre_cnt, code_pre_right, code_true_cnt


def filter_grads_vals(ori_gv, condition):
    new_grads_vals = []
    for (g, v) in ori_gv:
        if condition(v):
            new_grads_vals.append((g, v))
    return new_grads_vals


def map_content_to_index(lst: list) -> dict:
    """
    把一个list转换为其值和和index下标的对应
    :param lst: 待转换的list，要求值不重复，否则装换没意义
    :return: dict
    """
    ret = dict()
    for i, w in enumerate(lst):
        ret[w] = i
    return ret


def reade_fn_as_label(file_data, corpus_dir, label_indicator='label'):
    """

    :param file_data:
    :param corpus_dir:
    :return:
    """
    cnt = 0
    summary_fn = os.path.join(corpus_dir, "summary.json")
    this_fn = os.path.basename(file_data)
    if os.path.exists(summary_fn):
        summary_dict = json.load(open(summary_fn))
    else:
        summary_dict = {"Summary": {"total": 0}}
    summary_dict[this_fn] = {"total": 0}

    with open(file_data, 'r', encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            line = line.strip()
            if len(line) == 0:
                print("empty line")
                continue
            cnt += 1
            if i % 2000 == 0:
                print("cnt={}".format(cnt))
            try:
                tmp_json = json.loads(line)
            except:
                print(line)
            label = tmp_json[label_indicator]
            if label not in summary_dict["Summary"]:
                summary_dict["Summary"][label] = 0
            if label not in summary_dict[this_fn]:
                summary_dict[this_fn][label] = 0

            summary_dict["Summary"][label] += 1
            summary_dict["Summary"]["total"] += 1
            summary_dict[this_fn][label] += 1
            summary_dict[this_fn]["total"] += 1

            yield tmp_json, label

    print("total cnt={}".format(cnt))

    with open(summary_fn, 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(summary_dict))

    print("[File] saving corpus summary into %s" % summary_fn)


def padding(sent, max_len, eos, pad):
    """
    pad a sentence to seqcified length using eos and pad.
    sometimes we can use eos==pad
    :param sent: list_of_string
    :param max_len:  int
    :param eos:  dictionary.EOS
    :param pad:  dictionary.PAD
    :return:
    """
    if not isinstance(sent, list):
        sent = list(sent)

    if len(sent) >= max_len:
        if sent[max_len - 1] in [pad, eos]:    # 避免重复 padding (20180829)
            return sent[:max_len]
        else:
            return sent[:max_len - 1] + [eos]
    return sent + [eos] + [pad] * (max_len - len(sent) - 1)