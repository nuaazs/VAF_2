# 绑定的主机和端口
bind = "0.0.0.0:8989"

# Gunicorn 的工作进程数
workers = 4

# 每个工作进程的线程数
threads = 2

module = "main:app"

# 访问日志文件的路径
accesslog = "/log/gunicorn/access.log"

# 错误日志文件的路径
errorlog = "/log/gunicorn/error.log"

# 是否以守护进程方式运行（后台运行）
daemon = True

worker_class = "gevent"  # 使用gevent模式，还可以使用sync 模式，默认的是sync模式

pidfile = "/log/gunicorn/gunicorn.pid"  # 设置进程文件的路径