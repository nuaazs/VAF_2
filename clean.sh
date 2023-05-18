#!/bin/bash
# remote pretrained_models __pycache__ and *.pyc

find . -name "__pycache__" | xargs rm -rf
find . -name "*.pyc" | xargs rm -rf
find . -name "pretrained_models" | xargs rm -rf
find . -name "*.log" | xargs rm