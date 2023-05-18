#!/bin/bash
gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn.py main:app --timeout 1000