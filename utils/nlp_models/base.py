import re
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

lemmatizer = WordNetLemmatizer()
punc_words = list('\',.?!:;"')


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return wn.NOUN


def clean_tweet(tweet):
    # 去掉印度英文的tweet
    if tweet.find(' ka ') >= 0 or tweet.find(' se ') >= 0:
        return ''
    # remove \uxxxx
    tmp = tweet.replace('\u2019', "'").lower().replace('\xa0', ' ').replace('\u2014', '').replace(" $ ", '')
    # 去掉http：// pic.连接
    tmp = re.sub("(http.*\u2026)|(http.*\/[\S]+)|(pic\.twitter.*\/[\S]+)", "", tmp)
    tmp = re.sub("\$[\d\.]+", "dollar", tmp)
    tmp = re.sub('#\w+', '', tmp)
    # 去掉所有cashtag
    tmp = re.sub("\$\w+", "", tmp)
    tmp = tmp.replace('@', '')
    # tmp = tmp.replace('\'s', '').replace('u.s.', ' us ')
    tmp = tmp.replace("\\'", '').replace('\\', '')
    tmp = tmp.replace('\n', ' ')
    for punc in punc_words:
        tmp = tmp.replace(punc, '')

    tokens = tmp.split(' ')
    lst_tweet = []
    for token in tokens:
        if token == '':
            continue
        lst_tweet.append(token)
    if len(lst_tweet) < 10:
        return ''
    pos_val = nltk.pos_tag(lst_tweet)
    lemma_words = [lemmatizer.lemmatize(w, penn_to_wn(tag)) for (w, tag) in pos_val]
    return ' '.join(lemma_words)


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