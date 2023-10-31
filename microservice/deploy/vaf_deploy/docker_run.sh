#!/bin/bash

image_name="vaf_server:v1.0"
container_name="vaf_server"

if [ -n "$(sudo docker ps -a | grep $container_name)" ]; then
    echo "Remove old container: $container_name"
    sudo docker rm -f $container_name
    echo sudo docker rm -f $container_name
fi

# --gpus '"device=0,1"'
sudo docker run --gpus all --restart=always --network="host" -d -it \
 --name $container_name \
 --entrypoint="" \
$image_name bash
