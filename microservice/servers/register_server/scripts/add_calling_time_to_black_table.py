#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   add_calling_time_to_black_table.py
@Time    :   2023/10/16 14:31:14
@Author  :   Carry
@Version :   1.0
@Desc    :   给黑名单表添加通话开始时间字段
'''


from datetime import datetime, timedelta
import random

# 输入的时间字符串
input_time_str = "2023-10-15 11:44:55"
input_time = datetime.strptime(input_time_str, "%Y-%m-%d %H:%M:%S")

# 随机生成 0 到 100 之间的整数作为时长
time_difference = timedelta(seconds=random.randint(0, 100))

# 添加时长
new_time = input_time + time_difference

# 格式化为字符串
new_time_str = new_time.strftime("%Y-%m-%d %H:%M:%S")

print(new_time_str)


