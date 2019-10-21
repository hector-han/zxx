import numpy as np
from utils.nlp_models.base import load_all_tweets, clean_tweet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scripts.export_mysql_file import build_tweet_content





if __name__ == '__main__':
    tweet_file = r'D:\code\github\data\all_tweets.jl'
    tweet_sql_file = r'D:\code\github\data\all_tweets_cate.sql'
    all_tweets = load_all_tweets(tweet_file)
    index_to_tweetid = []
    corpus_list = []
    tweetid_to_index = {}
    idx = 0
    for id, tweet in all_tweets.items():
        cleaned = clean_tweet(tweet['text'])
        if id in tweetid_to_index.keys():
            continue
        if cleaned == '':
            continue
        index_to_tweetid.append(id)
        corpus_list.append(cleaned)
        tweetid_to_index[id] = idx
        idx += 1
    print('begin count...')
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus_list)
    print('begin fit lda')
    lda = LatentDirichletAllocation(n_components=5,
                                    random_state=0)
    docres = lda.fit_transform(X)
    cate_labels = np.argmax(docres, axis=1)

    tweets_id_saved = set()
    print('writing file')
    with open(tweet_sql_file, 'w', encoding='utf-8') as fout:
        for id, tweet in all_tweets.items():
            if id not in tweets_id_saved:
                cate = -1
                cleaned = ''
                if id in tweetid_to_index:
                    idx = tweetid_to_index[id]
                    cate = cate_labels[idx]
                    cleaned = corpus_list[idx]
                _, content = build_tweet_content(tweet, extra=[str(cate), cleaned])
                fout.write(content)
                fout.write('\n')
                tweets_id_saved.add(id)









