# coding = utf-8
# @Time    : 2022-09-05  09:46:15
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Gunicorn config file.

import cfg

# allow multithreading in gunicorn
threaded = True
workers = cfg.WORKERS
threads = cfg.THREADS
bind = f"0.0.0.0:{cfg.PORT}"
daemon = "true"
# worker_class in ：sync, eventlet, gevent, tornado, gthread
worker_class = "tornado"
worker_connections = cfg.WORKER_CONNECTIONS
pidfile = "./log/gunicorn.pid"
accesslog = "./log/gunicorn_access.log"
errorlog = "./log/gunicorn_error.log"
loglevel = "info"
