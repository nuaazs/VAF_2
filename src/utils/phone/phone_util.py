# coding = utf-8
# @Time    : 2022-09-05  15:12:04
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Phone.

from phone import Phone


def get_phone_info(phoneNum):
    info = Phone().find(phoneNum)
    if info == None:
        return {}
    return info
