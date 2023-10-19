#!/bin/bash

APP_NAME="app"
PID_FILE="$APP_NAME.pid"

# 检查应用是否在运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    echo "正在停止应用 $APP_NAME (PID: $PID)"
    kill -15 $PID  # 发送SIGTERM信号以优雅地停止应用
    rm $PID_FILE
    echo "应用 $APP_NAME 已停止"
else
    echo "应用 $APP_NAME 未在运行"
fi
