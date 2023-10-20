#!/bin/bash

GUNICORN_CONF="gunicorn_config.py"
APP_MODULE="main:app"

check_app_availability() {
    HEALTH_CHECK_URL="http://localhost:5550/health"

    HTTP_STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_CHECK_URL | tr -d '\n')

    if [ "$HTTP_STATUS_CODE" -eq 200 ]; then
        echo 0
    else
        echo 1
    fi
}

# 检查当前目录是否有log文件夹，没有则创建
if [ ! -d "log" ]; then
    mkdir log
fi
result=$(check_app_availability)
if [ "$result" -eq 0 ]; then
    echo "Gunicorn server is running"
else
    echo "Gunicorn server is not running."
    echo "Starting Gunicorn server..."
    gunicorn -c $GUNICORN_CONF $APP_MODULE &
    echo "Gunicorn server is started."
fi