#!/bin/bash
echo "start docker build..."
docker build -t auto_test:v1.1 .
sleep 2
echo "end docker"
