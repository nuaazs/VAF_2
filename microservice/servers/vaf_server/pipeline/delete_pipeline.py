#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   encode_pipeline.py
@Time    :   2023/10/14 22:12:44
@Author  :   Carry
@Version :   1.0
@Desc    :   删除黑库数据流程
'''

import pymysql
import cfg
from tools.orm_handler import delete_by_key

def delete_pipeline(request):
    """
    删除黑库数据流程
    """
    spkid = request.form.get('spkid')
    if not spkid:
        return {"code": 500, "message": "spkid is None."}
    conn = pymysql.connect(
        host=cfg.MYSQL.get("host"),
        port=cfg.MYSQL.get("port"),
        db=cfg.MYSQL.get("db"),
        user=cfg.MYSQL.get("username"),
        passwd=cfg.MYSQL.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    table_name = cfg.MYSQL.get("black_table_name")
    query = f'delete from {table_name} where record_id={spkid};'
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()

    delete_by_key(spkid)
    return {"code": 200, "message": "Delete success", "spkid": spkid}
