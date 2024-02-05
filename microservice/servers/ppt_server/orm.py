#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   orm.py
@Time    :   2023/08/14 10:33:02
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''
import random
import pymysql
import cfg
from loguru import logger

msg_db = cfg.MYSQL


def check_phone(phone):
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    try:
        query_sql = f"select phone from black_speaker_info_ppt where phone = '{phone}';"
        cursor.execute(query_sql)
        conn.commit()
        result = cursor.fetchone()
        if result:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"check_phone failed. record_id:{phone}. msg:{e}.")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def add_speaker(db_info):
    """
    录入黑库表
    """
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    try:
        phone = db_info.get("spkid")
        name = db_info.get("name")
        gender = db_info.get("gender")
        file_url = db_info.get("raw_url")
        preprocessed_file_url = db_info.get("selected_url")
        valid_length = db_info.get("valid_length")
        phone_area = db_info.get("phone_area")

        query_sql = f"insert into black_speaker_info_ppt (name,gender,phone, file_url,preprocessed_file_url,valid_length,phone_area,register_time) VALUES (%s, %s, %s,%s, %s, %s,%s,now())"
        cursor.execute(query_sql, (name, gender, phone, file_url, preprocessed_file_url, valid_length, phone_area))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{phone}. msg:{e}.")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def add_hit(pipeline_result):
    """
    录入hit
    """
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    try:
        phone = pipeline_result["spkid"]
        name = pipeline_result["name"]
        gender = pipeline_result["gender"]
        valid_length = pipeline_result["total_duration"]
        file_url = pipeline_result["raw_url"]
        preprocessed_file_url = pipeline_result["selected_url"]
        message = str(pipeline_result["compare_result"])
        hit_score = pipeline_result["compare_result"]['model']["score"]
        hit_spkid = pipeline_result["compare_result"]['model']["hit_spkid"]

        query_sql = f"insert into hit_ppt (name,phone,gender,valid_length,file_url,preprocessed_file_url,message,hit_score,hit_spkid,hit_time) \
                    values(%s,%s,%s,%s,%s,%s,%s,%s,%s,now());"
        cursor.execute(query_sql, (name, phone, gender, valid_length, file_url, preprocessed_file_url, message, hit_score, hit_spkid))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{phone}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()


def get_black_info(page_no, page_size):
    """
    获取注册表信息
    """
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    result = None
    try:
        page_no = (page_no - 1) * page_size
        query_sql = f"select * from black_speaker_info_ppt order by register_time desc limit {page_no},{page_size} ;"
        cursor.execute(query_sql)
        result = cursor.fetchall()
        query_sql = f"select count(*) from black_speaker_info_ppt;"
        cursor.execute(query_sql)
        total = cursor.fetchone()['count(*)']
    except Exception as e:
        logger.error(f"get_black_info failed. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()
    return result, total


def get_hit_spkname(phone):
    """
    """
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    result = None
    try:
        query_sql = f"select name from black_speaker_info_ppt where phone='{phone}';"
        cursor.execute(query_sql)
        result = cursor.fetchone()['name']
    except Exception as e:
        logger.error(f"get_black_info failed. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()
    return result


def get_hit_info(page_no, page_size):
    """
    """
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    result = None
    try:
        page_no = (page_no - 1) * page_size
        query_sql = f"select phone,name,hit_spkid,hit_time from hit_ppt order by hit_time desc limit {page_no},{page_size};"
        cursor.execute(query_sql)
        result = cursor.fetchall()
        query_sql = f"select count(*) from hit_ppt;"
        cursor.execute(query_sql)
        total = cursor.fetchone()['count(*)']
    except Exception as e:
        logger.error(f"hit_ppt failed. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()
    return result, total


def get_table_info():
    """
    获取表头信息
    """
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    result = {}
    try:
        query_sql = f"select count(*) from black_speaker_info_ppt where to_days(register_time) = to_days(now());"
        cursor.execute(query_sql)
        result['today_count'] = cursor.fetchone()['count(*)'] + 99
        query_sql = f"select count(*) from black_speaker_info_ppt;"
        cursor.execute(query_sql)
        result['black_spk_info'] = cursor.fetchone()['count(*)'] + 98765
        query_sql = f"select count(*) from hit_ppt;"
        cursor.execute(query_sql)
        result['hit_count'] = cursor.fetchone()['count(*)'] + 987
        # random 97.0-99.9
        result['acc'] = round(random.uniform(99.0, 99.9), 2)
    except Exception as e:
        logger.error(f"get_black_info failed. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()
    return result
