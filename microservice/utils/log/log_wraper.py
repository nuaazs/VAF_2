# coding = utf-8
# @Time    : 2022-09-05  15:05:31
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Define log wrapers.

from logging.handlers import RotatingFileHandler
import logging

# common log
logger = logging.getLogger("si_log")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s]  %(levelname)s  [%(filename)s]  #%(lineno)d %(filename)s:%(lineno)s - %(funcName)20s() <%(process)d:%(thread)d>  %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
handler = RotatingFileHandler(
    "./log/si.log", maxBytes=1 * 1024 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
handler.setFormatter(formatter)
handler.namer = lambda x: "si." + x.split(".")[-1]
logger.addHandler(handler)
# also print error log to console
# console_ = logging.StreamHandler()
# console_.setLevel(logging.DEBUG)
# console_.setFormatter(formatter)
# logger.addHandler(console_)

# error log
err_logger = logging.getLogger("err_log")
err_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s]  %(levelname)s  [%(filename)s]  #%(lineno)d %(filename)s:%(lineno)s - %(funcName)20s() <%(process)d:%(thread)d>  %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
err_handler = RotatingFileHandler(
    "./log/error.log", maxBytes=1 * 1024 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
err_handler.setFormatter(formatter)
err_handler.namer = lambda x: "err." + x.split(".")[-1]
err_logger.addHandler(err_handler)
# also print error log to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
err_logger.addHandler(console)
