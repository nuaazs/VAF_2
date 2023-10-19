#!/bin/bash

APP_NAME="app"
CONFIG_FILE="gunicorn_config.py"
PID_FILE="$APP_NAME.pid"

# 检查应用是否已经在运行
if [ -f "$PID_FILE" ]; then
    echo "应用 $APP_NAME 已经在运行 (PID: $(cat $PID_FILE))"
    exit 1
fi

# 启动 Gunicorn 应用
gunicorn -c $CONFIG_FILE $APP_NAME:app -D

# 获取新的应用进程的PID并保存到PID文件
echo $! > $PID_FILE
echo "应用 $APP_NAME 已启动 (PID: $(cat $PID_FILE))"
