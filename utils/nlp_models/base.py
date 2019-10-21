import re
import json

def clean_tweet(tweet):
    # remove \uxxxx
    tmp = tweet.replace('\u2019', "'").lower()
    tmp = re.sub("(http.*\u2026)|(http.*\/[\S]+)|(pic\.twitter.*\/[\S]+)", "", tmp)
    lst_tweet = re.sub("(@[A-Za-z0-9]+)|([^A-Za-z0-9 '\t])", " ", tmp).split()
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