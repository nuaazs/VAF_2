# coding = utf-8
# @Time    : 2022-09-05  15:06:29
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Query in SQL.

import re
import pymysql
import time
from utils.log import logger
import cfg
from utils.orm.db_utils import mysql_handler, call_time

msg_db = cfg.MYSQL


def get_span(time2, time1):
    time2 = time.strptime(time2, "%Y-%m-%d %H:%M:%S")
    time1 = time.strptime(time1, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time2) - time.mktime(time1))


def check_url(url):
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    while True:
        try:
            cur = conn.cursor()
            query_sql = f"SELECT * FROM speaker WHERE file_url='{url}';"
            cur.execute(query_sql)
            res = cur.fetchall()
            if len(res) != 0:
                conn.close()
                return True
            else:
                conn.close()
                return False
        except Exception as error:
            conn.ping(True)


def get_wav_url(spk_id):
    conn = pymysql.connect(
        host=msg_db.get("host", "zhaosheng.mysql.rds.aliyuncs.com"),
        port=msg_db.get("port", 27546),
        db=msg_db.get("db", "si"),
        user=msg_db.get("user", "root"),
        passwd=msg_db.get("passwd", "Nt3380518!zhaosheng123"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    query_sql = f"select file_url from speaker where phone='{spk_id}' and status=1 limit 1;"
    cur.execute(query_sql)
    result = cur.fetchall()
    if len(result) > 0:
        conn.commit()
        conn.close()
        return result[0]["file_url"]
    else:
        conn.commit()
        conn.close()
        return None


def get_spkinfo(spk_id):
    conn = pymysql.connect(
        host=msg_db.get("host", "zhaosheng.mysql.rds.aliyuncs.com"),
        port=msg_db.get("port", 27546),
        db=msg_db.get("db", "si"),
        user=msg_db.get("user", "root"),
        passwd=msg_db.get("passwd", "Nt3380518!zhaosheng123"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    query_sql = f"select class_number,self_test_score_mean,valid_length,register_time from speaker where phone='{spk_id}' and status=1 limit 1;"
    cur.execute(query_sql)
    result = cur.fetchall()
    if len(result) > 0:
        conn.commit()
        conn.close()
        return {
            "class_number": result[0]["class_number"],
            "self_test_score_mean": result[0]["self_test_score_mean"],
            "valid_length": result[0]["valid_length"],
            "register_time": result[0]["register_time"],
        }
    else:

        conn.commit()
        conn.close()
        return None


def delete_spk(spk_id):
    conn = pymysql.connect(
        host=msg_db.get("host", "zhaosheng.mysql.rds.aliyuncs.com"),
        port=msg_db.get("port", 27546),
        db=msg_db.get("db", "si"),
        user=msg_db.get("user", "root"),
        passwd=msg_db.get("passwd", "Nt3380518!zhaosheng123"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    query_sql = f"delete from speaker where phone='{spk_id}';"
    print(f"delete from speaker where phone='{spk_id}';")
    cur.execute(query_sql)
    result = cur.fetchall()
    if len(result) > 0:
        print(result)
    conn.commit()
    conn.close()


@call_time
def to_log_bak(
        phone,
        action_type,
        err_type,
        message,
        file_url,
        show_phone,
        preprocessed_file_path="",
        valid_length=0,
):
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    # todo 添加showphone
    conn.ping(reconnect=True)
    cur = conn.cursor()

    date_num = int(time.strftime("%d", time.localtime()))

    query_sql = f"INSERT INTO log_{date_num} (phone,show_phone,action_type,time,err_type, message,file_url,preprocessed_file_url) VALUES ('{phone}','{show_phone}','{action_type}', curtime(),'{err_type}', '{message}','{file_url}','{preprocessed_file_path}');"

    cur.execute(query_sql)
    conn.commit()
    conn.close()


@call_time
def add_hit_bak(hit_info, is_grey, after_vad_length):
    conn = pymysql.connect(
        host=msg_db.get("host", "zhaosheng.mysql.rds.aliyuncs.com"),
        port=msg_db.get("port", 27546),
        db=msg_db.get("db", "si"),
        user=msg_db.get("user", "root"),
        passwd=msg_db.get("passwd", "Nt3380518!zhaosheng123"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    phone = hit_info["phone"]
    show_phone = hit_info["show_phone"]
    file_url = hit_info["file_url"]
    #
    province = hit_info["province"]
    city = hit_info["city"]
    phone_type = hit_info["phone_type"]
    area_code = hit_info["area_code"]
    zip_code = hit_info["zip_code"]
    self_test_score_mean = hit_info["self_test_score_mean"]
    self_test_score_min = hit_info["self_test_score_min"]
    self_test_score_max = hit_info["self_test_score_max"]
    call_begintime = hit_info["call_begintime"]
    call_endtime = hit_info["call_endtime"]
    valid_length = after_vad_length
    class_number = hit_info["class_number"]
    hit_time = hit_info["hit_time"]
    blackbase_phone = hit_info["blackbase_phone"]
    blackbase_id = hit_info["blackbase_id"]
    top_10 = hit_info["top_10"]
    # 1~10
    hit_status = hit_info["hit_status"]
    hit_score = hit_info["hit_scores"]
    preprocessed_file_path = hit_info["preprocessed_file_path"]
    if is_grey:
        is_grey = 1
    else:
        is_grey = 0
    cur = conn.cursor()
    query_sql = f"INSERT INTO hit (phone, file_url, phone_type,area_code,\
                    self_test_score_mean,self_test_score_min,self_test_score_max,call_begintime,\
                    call_endtime,valid_length,class_number,blackbase_phone,blackbase_id,top_10,hit_status,hit_score,preprocessed_file_url,is_grey,show_phone,hit_time) \
                        VALUES ('{phone}', '{file_url}','{phone_type}','{area_code}',\
                    '{self_test_score_mean}','{self_test_score_min}','{self_test_score_max}','{call_begintime}',\
                    '{call_endtime}','{valid_length}','{class_number}','{blackbase_phone}','{blackbase_id}','{top_10}',\
                    '{hit_status}','{hit_score}','{preprocessed_file_path}','{is_grey}','{show_phone}',NOW());"
    try:
        cur.execute(query_sql)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(e)


@call_time
def add_speaker_bak(spk_info, after_vad_length):
    conn = pymysql.connect(
        host=msg_db.get("host", "zhaosheng.mysql.rds.aliyuncs.com"),
        port=msg_db.get("port", 27546),
        db=msg_db.get("db", "si"),
        user=msg_db.get("user", "root"),
        passwd=msg_db.get("passwd", "Nt3380518!zhaosheng123"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    name = spk_info["name"]
    phone = spk_info["phone"]
    file_url = spk_info["uuid"]
    register_time = spk_info["register_time"]
    province = spk_info["province"]
    city = spk_info["city"]
    phone_type = spk_info["phone_type"]
    area_code = spk_info["area_code"]
    zip_code = spk_info["zip_code"]
    self_test_score_mean = spk_info["self_test_score_mean"]
    self_test_score_min = spk_info["self_test_score_min"]
    self_test_score_max = spk_info["self_test_score_max"]
    call_begintime = spk_info["call_begintime"]
    call_endtime = spk_info["call_endtime"]
    class_number = spk_info["max_class_index"]
    preprocessed_file_path = spk_info["preprocessed_file_path"]
    show_phone = spk_info["show_phone"]
    valid_length = after_vad_length
    cur = conn.cursor()
    query_sql = f"INSERT INTO speaker (name,phone, file_url,phone_type,area_code,\
                    self_test_score_mean,self_test_score_min,self_test_score_max,call_begintime,\
                    call_endtime,valid_length,class_number,preprocessed_file_url,show_phone,register_time) \
                        VALUES ('{name}','{phone}', '{file_url}', '{phone_type}','{area_code}',\
                    '{self_test_score_mean}','{self_test_score_min}','{self_test_score_max}','{call_begintime}',\
                    '{call_endtime}','{valid_length}','{class_number}','{preprocessed_file_path}','{show_phone}',NOW());"
    try:
        cur.execute(query_sql)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(e)


@call_time
def add_hit_count_bak(spk_id):
    conn = pymysql.connect(
        host=msg_db.get("host", "zhaosheng.mysql.rds.aliyuncs.com"),
        port=msg_db.get("port", 27546),
        db=msg_db.get("db", "si"),
        user=msg_db.get("user", "root"),
        passwd=msg_db.get("passwd", "Nt3380518!zhaosheng123"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    query_sql = (
        f"update speaker set hit_count = hit_count + 1 where phone='{spk_id}' limit 1;"
    )
    try:
        cur.execute(query_sql)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(e)


@call_time
def get_blackid_bak(blackbase_phone):
    conn = pymysql.connect(
        host=msg_db.get("host", "zhaosheng.mysql.rds.aliyuncs.com"),
        port=msg_db.get("port", 27546),
        db=msg_db.get("db", "si"),
        user=msg_db.get("user", "root"),
        passwd=msg_db.get("passwd", "Nt3380518!zhaosheng123"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()
    query_sql = f"select id from speaker where phone='{blackbase_phone}' limit 1;"
    cur.execute(query_sql)
    result = cur.fetchall()
    if len(result) > 0:
        conn.close()
        return result[0]["id"]
    else:
        conn.close()
        return 0


@call_time
def check_spkid(spkid, action=cfg.DUPLICATE_TYPE):
    # TODO: 添加判断是否需要更新声纹
    # 如果有老数据，则删除
    try:
        query_sql = f"SELECT * FROM speaker WHERE phone='{spkid}' AND status=1;"
        result = mysql_handler.fetch_one(query_sql)
        if result is not None:
            return True
        else:
            # 如果有status 为0的数据
            if action == "remove_old":
                query_sql = f"SELECT * FROM speaker WHERE phone='{spkid}' AND status=0;"
                result_del = mysql_handler.fetch_one(query_sql)
                if result_del is not None:
                    del_sql = f"DELETE FROM speaker WHERE phone='{spkid}' AND status=0;"
                    mysql_handler.delete(del_sql)
                return False
    except Exception as e:
        logger.error(e)


@call_time
def add_speaker(spk_info, after_vad_length):
    name = spk_info["name"]
    phone = spk_info["phone"]
    file_url = spk_info["uuid"]
    register_time = spk_info["register_time"]
    province = spk_info["province"]
    city = spk_info["city"]
    phone_type = spk_info["phone_type"]
    area_code = spk_info["area_code"]
    zip_code = spk_info["zip_code"]
    call_begintime = spk_info["call_begintime"]
    call_endtime = spk_info["call_endtime"]
    class_number = spk_info["max_class_index"]
    preprocessed_file_path = spk_info["preprocessed_file_path"]
    show_phone = spk_info["show_phone"]
    valid_length = after_vad_length
    query_sql = f"INSERT INTO speaker (name,phone, file_url,phone_type,area_code,\
                    call_begintime,\
                    call_endtime,valid_length,class_number,preprocessed_file_url,show_phone,register_time) \
                        VALUES ('{name}','{phone}', '{file_url}', '{phone_type}','{area_code}',\
                    '{call_begintime}',\
                    '{call_endtime}','{valid_length}','{class_number}','{preprocessed_file_path}','{show_phone}',NOW());"
    try:
        mysql_handler.insert_one(query_sql)
    except Exception as e:
        logger.error(e)


@call_time
def add_hit_count(spk_id):
    query_sql = (
        f"update speaker set hit_count = hit_count + 1 where phone='{spk_id}' limit 1;"
    )
    mysql_handler.update(query_sql)


@call_time
def get_blackid(blackbase_phone):
    query_sql = f"select id from speaker where phone='{blackbase_phone}' limit 1;"
    result = mysql_handler.fetch_one(query_sql)
    return result.get("id", 0)


@call_time
def add_hit(hit_info, is_grey, after_vad_length):
    phone = hit_info["phone"]
    show_phone = hit_info["show_phone"]
    file_url = hit_info["file_url"]
    #
    province = hit_info["province"]
    city = hit_info["city"]
    phone_type = hit_info["phone_type"]
    area_code = hit_info["area_code"]
    zip_code = hit_info["zip_code"]
    call_begintime = hit_info["call_begintime"]
    call_endtime = hit_info["call_endtime"]
    valid_length = after_vad_length
    class_number = hit_info["class_number"]
    hit_time = hit_info["hit_time"]
    blackbase_phone = hit_info["blackbase_phone"]
    blackbase_id = hit_info["blackbase_id"]
    top_10 = hit_info["top_10"]
    # 1~10
    hit_status = hit_info["hit_status"]
    hit_score = hit_info["hit_scores"]
    preprocessed_file_path = hit_info["preprocessed_file_path"]
    content_text = hit_info["content_text"]
    hit_keyword = hit_info["hit_keyword"]
    keyword = hit_info["keyword"]

    if is_grey:
        is_grey = 1
    else:
        is_grey = 0

    query_sql = f"INSERT INTO hit (phone, file_url, phone_type,area_code,call_begintime,call_endtime,valid_length,class_number,\
                                   blackbase_phone,blackbase_id,top_10,hit_status,hit_score,preprocessed_file_url,\
                                   is_grey,show_phone,hit_time,hit_keyword,keyword,content_text) \
                 VALUES ('{phone}', '{file_url}','{phone_type}','{area_code}',\
                 '{call_begintime}',\
                 '{call_endtime}','{valid_length}','{class_number}','{blackbase_phone}','{blackbase_id}','{top_10}',\
                 '{hit_status}','{hit_score}','{preprocessed_file_path}','{is_grey}','{show_phone}',NOW(),'{hit_keyword}','{keyword[:1000]}','{content_text}');"
    try:
        mysql_handler.insert_one(query_sql)
    except Exception as e:
        logger.error(query_sql)
        logger.error(e)


@call_time
def to_log(
        phone,
        action_type,
        err_type,
        message,
        file_url,
        show_phone,
        preprocessed_file_path="",
        valid_length=0,
        before_length=0,
        after_length=0,
):
    date_num = int(time.strftime("%d", time.localtime()))
    query_sql = f"INSERT INTO log_{date_num} (phone,show_phone,action_type,time,err_type, message,file_url,\
                 preprocessed_file_url,before_length,after_length) VALUES ('{phone}','{show_phone}','{action_type}', curtime(),'{err_type}', \
                 '{message}','{file_url}','{preprocessed_file_path}',{before_length},{after_length});"
    try:
        logger.info(query_sql)
        mysql_handler.insert_one(query_sql)
    except Exception as e:
        logger.error(e)
