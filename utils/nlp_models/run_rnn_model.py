# -*- coding:utf-8 -*-
import os
import json
import tensorflow as tf
from utils.nlp_models.tools.constants import  VocabParams
from utils.nlp_models.general_rnn import RNNModel
from utils.nlp_models.tools.dictionary import CommonDictionary
from utils.nlp_models.tools.data_loader import DataLoader
from utils.nlp_models.tools.model_snippets import check_early_stop
from utils.nlp_models.tools.cus_logger import get_logger

"""
a bi-lstm implemation for short text classification using tensroflow library
"""

logger = get_logger(__name__)

class args:
    vocab_dir = '../../data/vocab'
    over_sample = None
    num_epochs = 100
    batch_size = 128
    checkpoint_every = 15
    patient = 150
    learning_rate = 0.001
    decay_rate = 0.95
    dropout_keep_prob = 0.6
    embedding_dim = 64
    num_hidden = 64
    active = ''
    train_file = '../../data/train.jl'
    valid_file = '../../data/valid.jl'
    test_file = '../../data/test.jl'
    log_dir = '../../data/tensorboard'
    ckpt_dir = '../../data/rnn_ckpt'



def train():
    logger.info("===============================")
    logger.info(args)
    logger.info("===============================")

    dictionary = CommonDictionary()
    dictionary.load_update_dictionary(dict_dir=args.vocab_dir)
    train_data_loader = DataLoader(
        filename=args.train_file,
        max_seq_len=dictionary.max_seq_len,
        mode="train",
        batch_size=args.batch_size)
    valid_data_loader = DataLoader(
        filename=args.valid_file,
        max_seq_len=dictionary.max_seq_len,
        mode="valid",
        batch_size=args.batch_size)

    rnn_model = RNNModel(dictionary=dictionary, args=args, graph_from_meta=None)

    with tf.Session() as sess:
        early_stop_cnt = 0
        rnn_model.initialize(sess)
        rnn_model.summary(graph=sess.graph, log_dir=args.log_dir)
        valid_acc = [1e-18]
        valid_losses = [1e18]
        i_epoch = train_data_loader.epoch
        while i_epoch < args.num_epochs:
            if check_early_stop(early_stop_cnt, args.patient):
                print('early stop!!!')
                break

            train_batch = train_data_loader.next_loop_batch(sess)
            rnn_model.step(sess, train_batch, epoch=i_epoch)
            i_epoch = train_data_loader.epoch
            global_step = rnn_model.global_step.eval()
            if global_step % args.checkpoint_every == 0:
                val_accuray, val_loss = rnn_model.evaluate(
                    sess, valid_data_loader=valid_data_loader, epoch=i_epoch, global_step=global_step)
                logger.info('val_accuray is:{}, val_loss is:{}'.format(val_accuray, val_loss))
                early_stop_cnt += 1
                if val_accuray >= valid_acc[-1]:
                    valid_acc.append(val_accuray)
                    valid_losses.append(val_loss)
                    rnn_model.save(sess, ckpt_dir=args.ckpt_dir, global_step=global_step)
                    early_stop_cnt = 0
                    logger.info("[Train] model improving and saved!")


def test():
    logger.info("=========================")
    dictionary = CommonDictionary(text_indicator='refine_user_acts')
    dictionary.load_update_dictionary(dict_dir=args.vocab_dir)
    dictionary.load_model(model_dir=args.vocab_dir)
    test_data_loader = DataLoader(
        filename=args.test_file,
        mode="test",
        dictionary=dictionary, prefix='')
    model_dir = args.save_dir
    if not tf.train.get_checkpoint_state(model_dir):
        raise  ValueError("must supply pre-train model when you are  testing!!!")

    result_fn = os.path.join(args.save_dir, "results.json")
    ckpt = tf.train.latest_checkpoint(checkpoint_dir=model_dir)
    meta_fn = ckpt + ".meta"
    logger.info("Reading pre-train model from %s"%model_dir)
    rnn_model = RNNModel(dictionary=dictionary, args=args, graph_from_meta=meta_fn)

    total_cnt = 0
    correct_cnt = 0

    with tf.Session() as sess:
        logger.info("Reading pre-train weights from %s" % model_dir)
        rnn_model.load(sess, model_dir)
        global_step = sess.run("global_step:0")

        with open(result_fn, "w", encoding="utf8") as fp:
            # fp.write("\t".join(["是否正确", "标注结果", "Top1预测结果", "预测分数", "句子"]) + "\n")
            for n_batch in range(test_data_loader.num_batches_per_epoch):
                batch_data = test_data_loader.next_batch(sess)
                pred_x_idx, topk_indices, topk_labels, topk_scores = rnn_model.inference(
                    sess, sentence_batch=batch_data["sentence"], seq_len=test_data_loader.max_seq_len,
                    length_batch=batch_data["length"]
                )
                for i, words in enumerate(batch_data["sentence"]):
                    call_id = batch_data["call_id"][i]
                    total_cnt += 1
                    true_label = str(batch_data["label"][i]).replace(VocabParams.label_prefix, "")
                    predict_label = topk_labels[i][0].decode("utf8").replace(VocabParams.label_prefix, "")
                    predict_score = topk_scores[i][0]
                    sentence = " ".join(words[:list(words).index(dictionary.PAD)])
                    label_flag = true_label == predict_label
                    if label_flag:
                        correct_cnt += 1
                    tmp_json = {"call_id": int(call_id), "true_label": true_label,
                                "predict_label": predict_label, "predict_score": float(predict_score),
                                "sentence": sentence}
                    fp.write(json.dumps(tmp_json) + "\n")
        acc = float(correct_cnt) / total_cnt if total_cnt != 0 else 0
        print("[results] %d / %d = %.4f" % (correct_cnt, total_cnt, acc))
        print("[results] %d / %d = %.4f" % (correct_cnt, total_cnt, acc))
        print("Top1 predict results saved into %s!" % result_fn)
        print("Done testing, model saved!")


if __name__ == "__main__":
    train()
    # test()