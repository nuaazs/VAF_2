# 绑定的主机和端口
bind = "0.0.0.0:5550"

# Gunicorn 的工作进程数
workers = 4

# 每个工作进程的线程数
threads = 2

# 访问日志文件的路径
accesslog = "log/access.log"

# 错误日志文件的路径
errorlog = "log/error.log"

worker_class = "gthread"  # 使用gevent模式，还可以使用sync 模式，默认的是sync模式,gthread

pidfile = "log/gunicorn.pid"  # 设置进程文件的路径

timeout = 600  # 超时