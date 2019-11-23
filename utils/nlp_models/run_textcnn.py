# --*-- coding: utf-8 --*--
import os
import sys
import json
import numpy as np
import tensorflow as tf

from utils.nlp_models.general_text_cnn import GeneralTextCNN
from utils.nlp_models.tools.entities import TrainOptions
from utils.nlp_models.tools.constants import VocabParams
from utils.nlp_models.tools.dictionary import CommonDictionary
from utils.nlp_models.tools.cus_logger import get_logger
from utils.nlp_models.tools.model_snippets import padding
from utils.nlp_models.tools.data_loader import DataLoader


class args:
    vocab_dir = '../../data/vocab'
    over_sample = None
    num_epochs = 100
    batch_size = 256
    checkpoint_every = 15
    patient = 20
    learning_rate = 0.001
    dropout_keep_prob = 0.6
    embedding_dim = 128
    filter_sizes = '2,3,4'
    num_filters = 16
    train_file = '../../data/train.jl'
    valid_file = '../../data/valid.jl'
    test_file = '../../data/test.jl'
    log_dir = '../../data/tensorboard'
    ckpt_dir = '../../data/cnn_ckpt'




logger = get_logger(__name__)
session_conf = tf.ConfigProto(allow_soft_placement=True)
mswindows = (sys.platform == "win32")


def recursive_translate(data, dict2idx: dict, default_value=None):
    if isinstance(data, (list, tuple)):
        ret = []
        for o in data:
            ret.append(recursive_translate(o, dict2idx, default_value))
        return ret
    if isinstance(data, str):
        if data in dict2idx:
            return dict2idx[data]
        else:
            return default_value


def read_data(data_file, dictionary: CommonDictionary):
    sentences = []
    labels = []
    with open(data_file, encoding='utf-8') as infile:
        for line in infile.readlines():
            try:
                line_json = json.loads(line)
            except json.JSONDecodeError:
                logger.error(line)
                continue
            label, text = line_json['label'], line_json['content']

            sentences.append(padding(sent=text, max_len=dictionary.max_seq_len, eos=VocabParams.eos, pad=VocabParams.pad))
            label = VocabParams.label_prefix + label
            labels.append(label)

    sentences_idx = recursive_translate(sentences, dictionary.w2index, dictionary.w2index[VocabParams.unk])
    labels_idx = recursive_translate(labels, dictionary.label2index, dictionary.label2index['__label__CENTRAL'])
    sentences_idx = np.asarray(sentences_idx, dtype=np.int32)
    labels_idx = np.asarray(labels_idx, dtype=np.int32)
    return sentences_idx, labels_idx


text_cnn_dictionary = CommonDictionary()
text_cnn_dictionary.load_update_dictionary(args.vocab_dir)

over_sample = None
if args.over_sample:
    over_sample = {}
    t = args.over_sample.split(",")
    for tt in t:
        ttt = tt.strip().split(':')
        over_sample[int(ttt[0].strip())] = int(ttt[1].strip())
print(over_sample)

train_options = TrainOptions(args.num_epochs, args.batch_size, args.checkpoint_every, args.patient,
                             args.dropout_keep_prob, over_sample)

label_size = text_cnn_dictionary.label_size
global_step = tf.Variable(0, name="global_step", trainable=False)

cnn_model = GeneralTextCNN(text_cnn_dictionary, train_options, global_step, learning_rate=args.learning_rate,
                           embedding_size=args.embedding_dim,
                           filter_sizes=list(map(int, args.filter_sizes.split(","))),
                           num_filters=args.num_filters)

saver = tf.train.Saver(max_to_keep=3)


def train():
    train_X, train_y = read_data(args.train_file, text_cnn_dictionary)
    valid_X, valid_y = read_data(args.valid_file, text_cnn_dictionary)
    print(valid_X)
    print(valid_y)
    n_samples = train_X.shape[0]
    cnn_new_features = np.zeros([n_samples, label_size - 1], dtype=np.float32)

    with tf.Session(config=session_conf) as sess:
        train_summary_dir = os.path.join(args.log_dir, "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        valid_summary_dir = os.path.join(args.log_dir, "valid")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        cnn_model.train(sess, train_X, train_y, valid_X, valid_y,
                        train_summary_writer, valid_summary_writer, saver, args.ckpt_dir)


def test():
    logger.info("===============================")
    data_loader = DataLoader(filename=args.test_file,
                                  max_seq_len=[400, 400],
                                  batch_size=1, mode="test")

    model_dir = args.ckpt_dir

    if not tf.train.get_checkpoint_state(model_dir):
        raise ValueError("must supply pre-train model when you are testing !!!")

    result_fn = os.path.join(model_dir, "results.json")
    ckpt = tf.train.latest_checkpoint(checkpoint_dir=model_dir)
    if not ckpt:
        logger.error("can not find ckpt file")
    logger.info("Reading pre-train model from %s" % model_dir)

    total_cnt = 0
    correct_cnt = 0
    with tf.Session(config=session_conf) as sess:
        logger.info("Reading pre-train weights from %s" % model_dir)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        cnn_model.initialize(sess)

        with open(result_fn, "w", encoding="utf8") as fp:
            while True:
                try:
                    batch_data = data_loader.one_shot_batch(sess)

                    feed_dict = {
                        cnn_model.input_x: batch_data['sentence'],
                    }

                    fetch_list = [
                        level2d_label_out, leve2d_model.soft_logits,
                        cnn_model.label_out, cnn_model.soft_logits,
                        fst_model.label_out, fst_model.soft_logits
                        ]
                    labels, all_scores, cnn_labels, cnn_scores, fst_labels, fst_scores = sess.run(fetch_list, feed_dict)
                    for i, words in enumerate(batch_data["raw"]):
                        call_id = batch_data["call_id"][i]
                        score = all_scores[i]
                        total_cnt += 1
                        true_label = str(batch_data["label"][i]).replace(VocabParams.label_prefix, "")
                        predict_label = labels[i].decode("utf8").replace(VocabParams.label_prefix, "")
                        cnn_label = cnn_labels[i].decode("utf8").replace(VocabParams.label_prefix, "")
                        fst_label = fst_labels[i].decode("utf8").replace(VocabParams.label_prefix, "")
                        cnn_score = cnn_scores[i]
                        fst_score = fst_scores[i]

                        label_flag = true_label == predict_label
                        if label_flag:
                            correct_cnt += 1
                        # tmp_list = [str(call_id), str(label_flag), true_label, predict_label, str(predict_score), sentence]
                        tmp_json = {"call_id": int(call_id), "raw": words,"true_label": true_label,
                                    "predict_label": predict_label, "all_score": score.tolist(),
                                    "cnn_label": cnn_label, "cnn_score": cnn_score.tolist(),
                                    "fst_label": fst_label, "fst_score": fst_score.tolist()}
                        fp.write(json.dumps(tmp_json) + "\n")
                except tf.errors.OutOfRangeError:
                    break

        acc = float(correct_cnt) / total_cnt if total_cnt != 0 else 0
        logger.info("[results] %d / %d = %.4f" % (correct_cnt, total_cnt, acc))
        logger.info("Top1 predict results saved into %s!" % result_fn)
        logger.info("Done testing, model saved!")



if __name__ == '__main__':
    train()
    # test()




