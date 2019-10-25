import numpy as np
import json
from scripts.settings import stop_words
from utils.nlp_models.base import load_all_tweets, clean_tweet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scripts.export_mysql_file import build_tweet_content



def build_model(raw_file, retfile):
    """
    :param raw_file:
    :param retfile:
    :return:
    """
    all_tweets = load_all_tweets(raw_file)
    index_to_tweetid = []
    corpus_list = []
    tweetid_to_index = {}
    idx = 0
    for id, tweet in all_tweets.items():
        cleaned = clean_tweet(tweet['text'])
        tweet['cleaned'] = cleaned
        if id in tweetid_to_index.keys():
            print('{} already exist')
            continue
        if cleaned == '':
            continue
        index_to_tweetid.append(id)
        corpus_list.append(cleaned)
        tweetid_to_index[id] = idx
        idx += 1
    print('begin count...')
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(corpus_list)
    print('begin fit lda')
    lda = LatentDirichletAllocation(n_components=3,
                                    random_state=0)
    docres = lda.fit_transform(X)

    tweets_id_saved = set()
    print('writing file')
    with open(retfile, 'w', encoding='utf-8') as fout:
        for id, tweet in all_tweets.items():
            if id not in tweets_id_saved:
                lda_score = []
                if id in tweetid_to_index:
                    idx = tweetid_to_index[id]
                    lda_score = docres[idx].tolist()
                #     cate = cate_labels[idx]
                #     cleaned = corpus_list[idx]
                # _, content = build_tweet_content(tweet, extra=[str(cate), cleaned])
                tweet['lda_score'] = lda_score
                fout.write(json.dumps(tweet))
                fout.write('\n')
                tweets_id_saved.add(id)


def export_mysql(infile, outfile):
    all_tweets = load_all_tweets(infile)
    with open(outfile, 'w', encoding='utf-8') as fout:
        for key, tweet in all_tweets.items():
            cleaned = tweet['cleaned'].replace('\n', '')[0:1999]
            cate = -1
            score = 0.0
            if len(tweet['lda_score']) > 0:
                cate = np.argmax(tweet['lda_score'])
                score = np.max(tweet['lda_score'])
            _, content = build_tweet_content(tweet, extra=[str(cate), str(score), cleaned])
            fout.write(content)
            fout.write('\n')


if __name__ == '__main__':
    tweet_file = r'D:\code\github\data\all_tweets_china.jl'
    tweet_sql_file = r'D:\code\github\data\all_tweets_cate_china_3.sql'
    tweet_lda_jlfile = r'D:\code\github\data\all_tweets_lda_3.jl'
    # build_model(tweet_file, tweet_lda_jlfile)
    export_mysql(tweet_lda_jlfile, tweet_sql_file)