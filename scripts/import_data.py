# --*-- encoding: utf-8 --*--

import sys
import os
import json
import logging
import pymysql
from settings import mysql_config


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
    ret = ret.replace('#,', '')[:-1]
    return ret


def build_tweet_sql(data):
    sql = """INSERT into `tweet` (`tweet_id`, `user_id`, `text`, `hash_tags`, `url`, `nbr_retweet`, `nbr_favorite`,
    `nbr_reply`, `datetime`, `has_media`, `medias`, `is_reply`, `is_retweet`) 
    VALUES ("{ID}", "{user_id}", "{text}", "{hash_tags}", "{url}", {nbr_retweet}, {nbr_favorite}, {nbr_reply}, 
    "{datetime}", {has_media}, "{medias}", {is_reply}, {is_retweet})
    """.format(**data)
    return sql


def build_user_sql(data):
    sql = """INSERT into `user` (`user_id`, `name`, `screen_name`, `avatar`) 
    VALUES ("{ID}", "{name}", "{screen_name}", "{avatar}")""".format(**data)
    return sql


def save_tweet(conn, data):
    with conn.cursor() as cursor:
        cursor.execute("select count(*) from tweet where tweet_id={}".format(data['ID']))
        row = cursor.fetchone()

    if row[0] > 0:
        logging.error('already have tweet, id={}'.format(data['ID']))
        cursor.close()
        return
    try:
        with conn.cursor() as cursor:
            for key, val in data.items():
                if type(val) == bool:
                    data[key] = int(val)
            data['hash_tags'] = extract_hash_tag(data['text'])
            if 'medias' not in data:
                data['medias'] = []
            if 'has_media' not in data:
                data['has_media'] = 0
            data['medias'] = ','.join(data['medias'])

            for key, val in data.items():
                if type(val) == str:
                    data[key] = conn.escape_string(val)
            sql = build_tweet_sql(data)
            cursor.execute(sql)
        conn.commit()
    except Exception as e:
        # 如果发生错误则回滚
        conn.rollback()
        logging.error(e)
        logging.error(sql)


def save_user(conn, data):
    with conn.cursor() as cursor:
        cursor.execute("select count(*) from user where user_id={}".format(data['ID']))
        row = cursor.fetchone()
    if row[0] > 0:
        logging.error('already have user, id={}'.format(data['ID']))
        cursor.close()
        return
    try:
        with conn.cursor() as cursor:
            for key, val in data.items():
                if type(val) == str:
                    data[key] = conn.escape_string(val)
            sql = build_user_sql(data)
            cursor.execute(sql)
        conn.commit()
    except Exception as e:
        # 如果发生错误则回滚
        conn.rollback()
        logging.error(e)
        logging.error(sql)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        logging.error("usage: import_data.py dirpath(eg 20180101-20180101)")
        exit(-1)
    dir = sys.argv[1]
    tweet_dir = os.path.join(dir, 'Data/tweet')
    if not os.path.exists(tweet_dir):
        logging.error('{}, no such file'.format(tweet_dir))
        exit(-1)
    user_dir = os.path.join(dir, 'Data/user')
    if not os.path.exists(user_dir):
        logging.error('{}, no such file'.format(user_dir))
        exit(-1)
    conn =  pymysql.connect(**mysql_config)
    for file in os.listdir(tweet_dir):
        if file.startswith('.'):
            continue
        with open(os.path.join(tweet_dir, file), encoding='utf-8') as fp:
            data = json.load(fp)
            save_tweet(conn, data)
    for file in os.listdir(user_dir):
        if file.startswith('.'):
            continue
        with open(os.path.join(user_dir, file), encoding='utf-8') as fp:
            data = json.load(fp)
            save_user(conn, data)
    conn.close()