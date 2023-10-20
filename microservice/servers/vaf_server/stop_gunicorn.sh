#!/bin/bash
GUNICORN_CONF="gunicorn_config.py"
APP_MODULE="main:app"

# 查找并终止 Gunicorn 进程
echo "Stopping Gunicorn server..."
pkill -f "gunicorn -c $GUNICORN_CONF $APP_MODULE"
echo "Gunicorn server is stopped."