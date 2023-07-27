#!/bin/bash
# stop gunicorn by pid
# pid saved in log/gunicorn.pid
_pid=$(cat log/gunicorn.pid)
kill -9 $_pid