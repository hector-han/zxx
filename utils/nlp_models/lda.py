import numpy as np
import json
import random
import pandas as pd
from scripts.settings import stop_words
from utils.nlp_models.base import load_all_tweets, clean_tweet
from scripts.export_mysql_file import build_tweet_content

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from pprint import pprint


punc_words = list('\',.?!:;')
punc_words.extend(stop_words)


def get_clean_tweet(raw_file, ret_file):
    """
    先清洗数据，保存起来。增加cleaned字段
    :param raw_file:
    :param ret_file:
    :return:
    """
    all_tweets = load_all_tweets(raw_file)
    cleand_tweets = {}
    unique_tweet = set()
    with open(r'D:\code\github\zxx\data\duplicated_clean', 'w', encoding='utf-8') as fout:
        for id, tweet in sorted(all_tweets.items(), key=lambda x: x[1]['datetime']):
            cleaned = clean_tweet(tweet['text'])
            if cleaned == '':
                continue
            if cleaned in unique_tweet:
                fout.write('already meet|{}|{}\n'.format(id, cleaned))
                continue
            unique_tweet.add(cleaned)
            tweet['cleaned'] = cleaned
            cleand_tweets[id] = tweet

    print('cnt={}, writing file'.format(len(cleand_tweets)))
    with open(ret_file, 'w', encoding='utf-8') as fout:
        for id, tweet in cleand_tweets.items():
            fout.write(json.dumps(tweet))
            fout.write('\n')


def build_model(raw_file, ret_file):
    """
    :param raw_file:
    :param retfile:
    :return:
    """
    all_tweets = load_all_tweets(raw_file)
    k = int(ret_file[ret_file.find('tweets_lda_') + 11])
    print('k={}'.format(k))
    idx2twetid = []
    common_texts = []
    for key, tweet in all_tweets.items():
        idx2twetid.append(key)
        tokens = tweet['cleaned'].split(' ')
        text = []
        for token in tokens:
            if token not in punc_words:
                text.append(token)
        common_texts.append(text)

    common_dictionary = Dictionary(common_texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    print('begin to train')
    lda_model = LdaModel(common_corpus, id2word=common_dictionary, num_topics=k, random_state=13)
    pprint(lda_model.print_topics(num_words=20))
    print('\nPerplexity: ', lda_model.log_perplexity(common_corpus))

    with open(ret_file, 'w', encoding='utf-8') as fout:
        for i, tweetid in enumerate(idx2twetid):
            tmp = lda_model[common_corpus[i]]
            lda_score = {}
            for ele in tmp:
                lda_score[str(ele[0])] = float(ele[1])
            all_tweets[tweetid]['lda'+str(k)] = lda_score
            fout.write(json.dumps(all_tweets[tweetid]))
            fout.write('\n')



def export_mysql(infile, outfile):
    all_tweets = load_all_tweets(infile)
    def get_cate_and_score(data):
        cate = ''
        maxscore = 0.0
        for key, val in data.items():
            if val > maxscore:
                maxscore = val
                cate = key
        return cate, maxscore

    # get label file
    cate2id = {}
    with open(outfile, 'w', encoding='utf-8', newline='\n') as fout:
        for key, tweet in all_tweets.items():
            cate, score = get_cate_and_score(tweet['lda5'])
            tweet['cate'] = cate
            tweet['score'] = score
            if cate not in cate2id:
                cate2id[cate] = []
            if tweet['choose']:
                cate2id[cate].append(key)

            _, content = build_tweet_content(tweet, extra=[str(cate), str(score), tweet['cleaned'].replace("\\", ""), tweet['senti']])
            fout.write(content)
            fout.write('\n')

    file_name = r'D:\code\github\zxx\data\情感标注.xlsx'
    excel_writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    total_cnt = sum([len(val) for val in cate2id.values()])
    for cate, idlist in cate2id.items():
        cnt = len(idlist)
        # sample_num = round(10000 * cnt / total_cnt)
        # print('sample {} for cate {}'.format(sample_num, cate))
        sample_list = idlist
        data_frame = pd.DataFrame([[id, all_tweets[id]['text'].replace('\n', ''), all_tweets[id]['cleaned']] for id in sample_list],
                                  columns=['tweetid', '原始推文', '清洗后推文'])
        data_frame.to_excel(excel_writer, sheet_name=cate)
    excel_writer.save()




if __name__ == '__main__':
    data_dir = r'D:\code\github\zxx\data'
    tweet_file = data_dir + r'\all_tweets_china.jl'
    cleand_file = data_dir + r'\all_tweets_cleaned.jl'
    # get_clean_tweet(tweet_file, cleand_file)

    tweet_lda_jlfile = data_dir + r'\all_tweets_lda_5.jl'
    # tweet_lda_jlfile = data_dir + r'\final.jl'
    # build_model(cleand_file, tweet_lda_jlfile)

    # tweet_senti_jlfile = data_dir + r'\all_tweets_senti.jl'
    tweet_senti_jlfile = data_dir + r'\all_tweets_vader_senti.jl'
    tweet_sql_file = data_dir + r'\all_tweets_vader_cate_5.sql'
    export_mysql(tweet_senti_jlfile, tweet_sql_file)