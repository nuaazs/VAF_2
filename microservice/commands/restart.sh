#!/bin/bash

function run {
    pid_file="log/gunicorn.pid"
    service_name="$1"

    if [ -f "$pid_file" ]; then
        echo "正在停止 $service_name 服务..."
        pid=$(cat "$pid_file")
        kill "$pid"
        echo "服务已停止。"
        # 删除pid文件
        rm "$pid_file"

        # start
        echo "正在重新启动 $service_name 服务..."
        gunicorn -c gunicorn.conf.py vaf_manager:app
    else
        echo "找不到 $service_name 服务的PID文件，服务可能未在运行中。"
    fi
}

