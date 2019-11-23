import pymysql
import datetime


from DBUtils.PersistentDB import PersistentDB
from scripts.settings import mysql_config, stop_words

db_pool = PersistentDB(
            # 指定数据库连接驱动
            creator=pymysql,
            # 一个连接最大复用次数,0或者None表示没有限制,默认为0
            maxusage=1000, **mysql_config)


def check_user(user_name, passwd):
    """检查用户密码是否正确"""
    db = db_pool.connection()
    sql = 'select * from t_accounts where name=%s'
    cursor = db.cursor()
    cursor.execute(sql, (user_name))
    data = cursor.fetchone()
    cursor.close()
    db.close()
    if not data:
        return False
    if data[0] == user_name and data[1] == passwd:
        return True
    else:
        return False


def get_next_task(current_user):
    """
    根据用户名获取下一条待标注记录
    :param current_user:
    :return:
    """
    db = db_pool.connection()
    sql = 'select `file_url`, `annotations` from t_audio_label where labeller=%s and label_cnt=0'
    cursor = db.cursor()
    cursor.execute(sql, (current_user))
    # (file_url, annotations)
    data = cursor.fetchone()
    cursor.close()
    db.close()
    return data


def update_label(file_url, annotations):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    db = db_pool.connection()
    sql = 'update t_audio_label set `annotations`=%s, `label_time`=%s, `label_cnt`=`label_cnt`+1 where `file_url`=%s'
    cursor = db.cursor()
    cursor.execute(sql, (annotations, now, file_url))
    db.commit()
    cursor.close()
    db.close()


def query_summary(start_time, end_time, cate='-2', sentiment="-1"):
    """
    根据起止时间查询每天的发布量
    :param start_time:
    :param end_time:
    :return:
    """
    if cate == '-2':
        sql = 'select date_format(datetime, "%%Y%%m%%d") as dt, count(1) as cnt ' \
              'from tweet where datetime between %s and %s '
    else:
        sql = 'select date_format(datetime, "%%Y%%m%%d") as dt, count(1) as cnt ' \
              'from tweet where datetime between %s and %s and topic={} '.format(cate)

    if sentiment != "-1":
        sql += " and sentiment='{}' ".format(sentiment)

    sql += "group by dt;"
    print(sql)
    db = db_pool.connection()
    dates = []
    values = []
    with db.cursor() as cursor:
        cursor.execute(sql, (start_time, end_time))
        rows = cursor.fetchall()
        if rows and len(rows) > 1:
            dates = [row[0] for row in rows]
            values = [row[1] for row in rows]
    db.close()
    return dates, values


def query_hash_tags(start_time, end_time, cate):
    """
    根据起止时间，查询出现的hashtag的次数
    :param start_time:
    :param end_time:
    :return:
    """
    column = 'clean_text'
    if cate == '-2':
        sql = 'select {} from tweet where datetime between %s and %s and {}<>"";'.format(column, column)
    else:
        sql = 'select {} from tweet where datetime between %s and %s and topic={} and {}<>"";'.format(column, cate, column)
    db = db_pool.connection()
    word_frequency = {}
    rows = None
    with db.cursor() as cursor:
        cursor.execute(sql, (start_time, end_time))
        rows = cursor.fetchall()
    db.close()
    if rows:
        for row in rows:
            for word in row[0].split(' '):
                word = word.replace('#', '').lower().strip()
                if word in stop_words:
                    continue
                if word not in word_frequency:
                    word_frequency[word] = 0
                word_frequency[word] += 1
    return word_frequency


def query_tweets_cnt(start_time, end_time, cate, sentiment="-1"):
    if cate == '-2':
        sql = 'select count(1) from tweet where datetime between %s and %s '
    else:
        sql = 'select count(1) from tweet where datetime between %s and %s and topic={} '.format(cate)

    if sentiment != "-1":
        sql += " and sentiment='{}';".format(sentiment)

    db = db_pool.connection()
    ret = None
    with db.cursor() as cursor:
        cursor.execute(sql, (start_time, end_time))
        ret = cursor.fetchone()
    db.close()
    if ret:
        return ret[0]
    return 0


tweets_fields = ['id', 'date_time', 'user_id', 'text', 'hash_tags', 'url', 'nbr_retweet', 'nbr_favorite', 'nbr_reply', 'sentiment']
def row_to_bootstraptable(row):
    result = {}
    for i, field in enumerate(tweets_fields):
        result[field] = row[i]
    return result


def query_tweets_list(start_time, end_time, limit, offset, sorted_by='nbr_retweet', cate='-2', sentiment="-1"):
    if cate == '-2':
        sql = 'select tweet_id, date_format(datetime, "%%Y%%m%%d") as df, user_id, text, hash_tags, url, nbr_retweet, nbr_favorite, nbr_reply, sentiment' \
          ' from tweet where datetime between %s and %s '
    else:
        sql = 'select tweet_id, date_format(datetime, "%%Y%%m%%d") as df, user_id, text, hash_tags, url, nbr_retweet, nbr_favorite, nbr_reply, sentiment' \
              ' from tweet where datetime between %s and %s and topic={} '.format(cate)
    if sentiment != "-1":
        sql += " and sentiment='{}' ".format(sentiment)

    sql += " order by {} desc limit {} offset {};".format(sorted_by, limit, offset)

    db = db_pool.connection()
    rows = []
    with db.cursor() as cursor:
        cursor.execute(sql, (start_time, end_time))
        rows = [row_to_bootstraptable(row) for row in cursor.fetchall()]
    db.close()
    return rows