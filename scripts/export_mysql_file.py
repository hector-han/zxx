import os
import json
import re


def filter_emoji(desstr,restr=''):
    #过滤表情
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)


def extract_hash_tag(text):
    ret = ''
    tagchar = False
    for c in text:
        if c == '#':
            ret += c
            tagchar = True
        elif tagchar:
            if c == ' ':
                ret += ','
                tagchar = False
            else:
                ret += c
    ret = ret.replace('\n', '')
    ret = ret.replace('#,', '')[:-1]
    return ret[:511]


def build_tweet_content(data):
    data['text'] = filter_emoji(data['text'])
    data['text'] = data['text'].replace('\\', '')
    for key, val in data.items():
        if type(val) == bool:
            data[key] = int(val)
    data['hash_tags'] = extract_hash_tag(data['text'])
    if 'medias' not in data:
        data['medias'] = []
    if 'has_media' not in data:
        data['has_media'] = 0
    data['medias'] = ','.join(data['medias'])

    return data['ID'], '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(data['ID'], data['user_id'], data['text'].replace('\n', ' ')[:1999], data['hash_tags'],
                                           data['url'], data['nbr_retweet'], data['nbr_favorite'], data['nbr_reply'],
                                           data['datetime'], data['has_media'], data['medias'], data['is_reply'],
                                           data['is_retweet'])


def build_user_content(data):
    data['name'] = filter_emoji(data['name'])
    data['name'] = data['name'].replace('\\', '')
    return data['ID'], '{}\t{}\t{}\t{}'.format(data['ID'], data['name'], data['screen_name'], data['avatar'])


if __name__ == '__main__':
    dir = r'D:\code\github\data'
    tweet_file = os.path.join(dir, 'all_tweets.jl')
    tweet_sql_file = os.path.join(dir, 'all_tweets.sql')
    user_file = os.path.join(dir, 'all_users.jl')
    user_sql_file = os.path.join(dir, 'all_users.sql')

    tweets_id_saved = set()
    with open(tweet_file, encoding='utf-8') as f, open(tweet_sql_file, 'w', encoding='utf-8') as fout:
        line = f.readline()
        while line:
            try:
                data = json.loads(line)
                id, content = build_tweet_content(data)
                if id not in tweets_id_saved:
                    fout.write(content)
                    fout.write('\n')
                    tweets_id_saved.add(id)
            except:
                print(line)
            line = f.readline()

    users_id_saved = set()
    with open(user_file, encoding='utf-8') as f, open(user_sql_file, 'w', encoding='utf-8') as fout:
        line = f.readline()
        while line:
            try:
                data = json.loads(line)
                id, content = build_user_content(data)
                if id not in users_id_saved:
                    fout.write(content)
                    fout.write('\n')
                    users_id_saved.add(id)
            except Exception as e:
                print(line)
            line = f.readline()