import re
import json
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
punc_words = list('\',.?!:;"')


def clean_tweet(tweet):
    # 去掉印度英文的tweet
    if tweet.find(' ka ') >= 0 or tweet.find(' se ') >=0:
        return ''
    # remove \uxxxx
    tmp = tweet.replace('\u2019', "'").lower().replace('\xa0', '').replace('-', ' ').replace(" $ ", '')
    # 去掉http：// pic.连接
    tmp = re.sub("(http.*\u2026)|(http.*\/[\S]+)|(pic\.twitter.*\/[\S]+)", "", tmp)
    tmp = re.sub("\$\d+", "$", tmp)
    # 去掉所有cashtag
    tmp = re.sub("\$[a-z]", "", tmp)
    tmp = tmp.replace('@', '').replace('#', '')
    tmp = tmp.replace('\'s', '').replace('u.s.', ' us ')
    tmp = tmp.replace("\\'", '')
    tmp = tmp.replace('\n', ' ')

    tokens = tmp.split(' ')
    lst_tweet = []
    for token in tokens:
        if token == '':
            continue
        elif token in punc_words:
            # lst_tweet.append(token)
            continue
        elif token[-1] in punc_words:
            # lst_tweet.append(lemmatizer.lemmatize(token[0:-1]))
            lst_tweet.append(token[0:-1])
            continue
        # lst_tweet.append(lemmatizer.lemmatize(token))
        lst_tweet.append(token)
    if len(lst_tweet) < 7:
        return ''
    return ' '.join(lst_tweet)


def load_all_tweets(tweet_file):
    """
    读去所有tweet
    :param path:
    :return:
    """
    all_tweets = {}
    with open(tweet_file, encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            all_tweets[data['ID']] = data
    return all_tweets