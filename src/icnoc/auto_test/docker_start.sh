#!/usr/bin/bash

image_name="auto_test:v1.1"

#无启动脚本
#sudo docker run --restart=always --network="host" -d -it \
#  -v /mnt:/mnt \
#  -v $(pwd)/config.py:/auto_test/config.py \
#  -v $(pwd)/logs:/auto_test/logs \
#  --name auto_test \
#  --entrypoint="" \
#  $image_name bash

#启动test
sudo docker run --restart=always --network="host" -d -it \
  -v /mnt:/mnt \
  -v $(pwd)/config.py:/auto_test/config.py \
  -v $(pwd)/logs:/auto_test/logs --name auto_test \
  --entrypoint /bin/bash $image_name -c "nohup python /auto_test/test.py > /auto_test/logs/progress_bars_test.txt 2>&1 & /bin/bash"

#启动main
#sudo docker run --restart=always --network="host" -d -it \
#  -v /datasets_hdd:/datasets_hdd \ 
#  -v $(pwd)/config.py:/auto_test/config.py \
#  -v $(pwd)/logs:/auto_test/logs \
#  --name auto_test \
#  $image_name
