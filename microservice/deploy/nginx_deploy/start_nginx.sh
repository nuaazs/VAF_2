#!/bin/bash

sudo docker rm -f nginx

sudo docker run -itd --name nginx -v ${PWD}/nginx.conf:/etc/nginx/nginx.conf -p 8787:8787 -p 8100:8100 nginx:latest 
echo "done"
