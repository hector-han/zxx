import pymysql
import datetime


from DBUtils.PersistentDB import PersistentDB
from scripts.settings import mysql_config

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