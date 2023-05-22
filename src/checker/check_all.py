# coding = utf-8
# @Time    : 2023-04-20  12:46:35
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Checker.

# import all checker from src.checker.*_checker.py
# and make a check_all function to check all the checker.

from checker.cuda_checker import check as check_cuda
from checker.minio_checker import check as check_minio
from checker.redis_checker import check as check_redis
from checker.mysql_checker import check as check_mysql
from checker.asr_checker import check as check_asr

def check_all():
    # check cuda
    cuda_check_result, cuda_check_message = check_cuda()
    # check minio
    minio_check_result, minio_check_message = check_minio()
    # check redis
    redis_check_result, redis_check_message = check_redis()
    # check mysql
    mysql_check_result, mysql_check_message = check_mysql()
    # check asr
    asr_check_result, asr_check_message = check_asr()
    # check all
    check_message_all = {
        "cuda": cuda_check_message,
        "minio": minio_check_message,
        "redis": redis_check_message,
        "mysql": mysql_check_message,
        "asr": asr_check_message,
    }
    check_result_all = {
        "cuda": cuda_check_result,
        "minio": minio_check_result,
        "redis": redis_check_result,
        "mysql": mysql_check_result,
        "asr": asr_check_result,
    }
    
    if cuda_check_result and minio_check_result and redis_check_result and mysql_check_result and asr_check_result:
        return True, check_result_all, check_message_all
    else:
        return False, check_result_all, check_message_all