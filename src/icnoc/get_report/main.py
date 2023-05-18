# -*- coding: utf-8 -*-
"""
@author: Carry
@file: main.py
@time: 2023/3/27 13:32
@desc: 获取近一轮测试的准确率报告，需要传入埋点号码到label.txt文件中

TP:预测是正样本，并且预测正确
FP:预测是正样本，并且预测错误
TN：预测是负样本，并且预测正确
FN：预测是负样本，并且预测错误

精确率：precision = TP / (TP + FP)
召回率：recall = TP / (TP + FN)
准确率：accuracy = (TP + TN) / (TP+ FP + TN + FN)
"""

import pymysql

conn = pymysql.connect(
    host="192.168.3.201",
    port=3306,
    db='si',
    user="root",
    passwd="longyuan",
    cursorclass=pymysql.cursors.DictCursor,
)
cur = conn.cursor()


def read_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data


def main():
    tp = fp = tn = fn = 0
    label_list = read_txt('./label.txt')
    label_list.pop(0)
    for i in label_list:
        phone = i.split(',')[0]
        black_id = i.split(',')[1].strip()
        query_sql = f"SELECT * FROM hit where phone = {phone} and hit_score > {hit_score}"
        cur.execute(query_sql)
        res = cur.fetchone()
        if res:
            if res['blackbase_phone'].split(',')[0] == black_id:
                tp += 1  # 预测正确
            else:
                fp += 1
        else:
            fn += 1  # 应该命中，但是没有命中

    tn = test_data_num - tp - fp - fn
    cur.close()
    conn.close()

    print(f"tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}")
    print(f"精确率：{tp / (tp + fp)}")
    print(f"召回率：{tp / (tp + fn)}")
    print(f"准确率：{(tp + tn) / (tp + fp + tn + fn)}")


def main_fun01():
    """
    遍历logs表，统计命中的数据
    Returns:

    """
    tp = fp = tn = fn = 0

    label_list = read_txt('./label.txt')
    label_list.pop(0)
    label_list = [i.split(',')[0] for i in label_list]

    query_sql = f"SELECT * FROM log_28"
    cur.execute(query_sql)
    res = cur.fetchall()
    for i in res:
        phone = i['phone']
        if phone in label_list:
            if 'True' in i['message']:
                tp += 1  # 预测正确
            elif 'False' in i['message']:
                fn += 1
        else:
            if 'True' in i['message']:
                fp += 1
            elif 'False' in i['message']:
                tn += 1

    cur.close()
    conn.close()

    print(f"tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}")
    print(f"精确率：{tp / (tp + fp)}")
    print(f"召回率：{tp / (tp + fn)}")
    print(f"准确率：{(tp + tn) / (tp + fp + tn + fn)}")


if __name__ == '__main__':
    hit_score = 0.78
    test_data_num = 50 * 10000
    # main()
    main_fun01()
