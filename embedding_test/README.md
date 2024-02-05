# Step 1
```shell
bash maidian.sh
```

# Step 2
```shell
bash acc.sh
```

注意：
1. Docker 容器：
	docker run -it -v /home/xz/duanyibo:/duanyibo -v /mnt:/mnt -v /opt/datasets:/datasets -v /opt/duanyibo_result:/result --name maidian --gpus "device=0" --net host zhaosheng/vaf_clean:v3.0
	需注意目录映射

