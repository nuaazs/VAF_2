## 1. 文件说明

```shell
# 假设文件保存在/home/deploy目录中
├─build
│  ├─html		 # html 服务代码及Dockerf
│  └─vaf		 # vaf 服务代码及Dockerfile
├─runtime
│  ├─html		 # html 服务配置文件及启动脚本
│  └─vaf		 # vaf  服务配置文件及启动脚本
└─clean          # vaf_clean 相关文件保存地址
```
设置主目录：
```shell
# 设置HOME_PATH
VAF_DEPLOY_HOME_PATH="/ssd2/icnoc_deploy_test/"
```



## 2. Docker 安装

略



## 3.安装启动nginx

```shell
# check md5
cd ${VAF_DEPLOY_HOME_PATH}/build/nginx_deploy
bash ./check.sh
#解压镜像
sudo docker load < nginx.tar
```
编辑配置文件nginx.conf 修改成相应的服务器地址及端口

```shell
vim nginx.conf (略)
```
启动服务

```shell
#修改相应的映射端口
#启动
bash ./start_nginx.sh
```

  

## 4.安装启动minio

```shell
# check md5
cd ${VAF_DEPLOY_HOME_PATH}/build/minio_deploy
bash ./check.sh
#修改相应配置文件(略)
vim ./start_minio.sh
#启动
bash ./start_minio.sh
```
以下按需配置：
登录http://IP:port登录后、手动添加桶




## 5. 离线安装redis

Ubuntu离线安装redis

```shell
# check md5
cd ${VAF_DEPLOY_HOME_PATH}/build/redis_deploy
bash ./check.sh
cd ./archives
sudo dpkg -i *.deb
sed -i "s/bind 127.0.0.1/bind 0.0.0.0/g" /etc/redis/redis.conf
sed -i "s/daemonize no/daemonize yes/g" /etc/redis/redis.conf
sudo redis-server /etc/redis/redis.conf
echo "done, please check with 'ps aux|grep redis'"
```



## 6. 离线安装mysql

```shell
# check md5
cd ${VAF_DEPLOY_HOME_PATH}/build/mysql_deploy
bash ./check.sh

#安装相关的deb
bash ./start.sh

#查看状态
service mysql status

#数据库相关配置
mysql -uroot -p

#修改密码及添加远程访问
create user 'test'@'%' identified with mysql_native_password by '123456';
select user,host from user;
grant all privileges on *.* to 'test'@'%';
flush privileges; 

#使用 test/123456 账号密码远程连接
```



## 7. VAF Clean Docker 安装

`vaf_clean:v2.0` 镜像包含了声纹服务系统的所有环境依赖。是`vaf`镜像的基础镜像。

tar包已经导出并进行了1G切片保存至`clean`目录下。

### 安装流程

1. 文件检查：核对md5码

   ```shell
   #!/bin/bash
   # check md5
   cd ${VAF_DEPLOY_HOME_PATH}/clean
   bash ./check.sh
   
   # cat splited files
   cat vaf_clean_v20_part_* > ${VAF_DEPLOY_HOME_PATH}/vaf_clean_v20.tar
   ```

2. 镜像载入：

   ```shell
   sudo docker load < ${VAF_DEPLOY_HOME_PATH}/vaf_clean_v20.tar
   ```

   

## 8. VAF Docker 

`vaf`是服务的核心镜像，包含了声纹编码模块等的相关代码。

### 安装流程

1. 文件检查：核对md5码

   ```shell
   # check md5
   cd ${VAF_DEPLOY_HOME_PATH}/build/vaf
   bash ./check.sh
   ```

2. 基于`vaf_clean`构建`vaf`镜像

    ```shell
    sudo docker build -t zhaosheng/vaf:v2.0 .
    ```

3. 数据库配置

    在数据库中新建`si`数据库，字符集`utf8`，排序规则`utf8_general_ci`,运行`database/si_0601.sql`文件。导入所需要的所有表单。

4. 修改配置文件

    ```shell
    cd ${VAF_DEPLOY_HOME_PATH}/runtime/vaf
    
    # vim cfg.py  修改mysql、redis、minio、服务端口等配置
    # vim ./start.sh  修改启动数量，取决于显卡数量
    
    # 启动服务
    sudo bash ./start.sh
    # 运行后新建 vaf_1 vaf_2 ... vaf_n 容器
    
    ########################################
    ################ vaf_n ################
    ########################################
    # 进入容器，并启动
    sudo docker exec -it vaf_1 bash
    # 进去容器后启动服务
    容器内> ./start.sh
    # 启动过程需输入密码，输入完成后ctrl+p+q退出前台
    # 重复上述过程，启动所有vaf，启动成功与否通过查看显存确认
    ```

    

## 9. HTML Docker

`html`包含了后端打分，黑库碰撞模块等的相关代码。

### 安装流程

1. 文件检查：核对md5码

   ```shell
   # check md5
   cd ${VAF_DEPLOY_HOME_PATH}/build/html
   bash ./check.sh
   ```

2. 基于`vaf_clean`构建`html`镜像。

   ```shell
   sudo docker build -t zhaosheng/html:v2.0 .
   ```
   
3. 修改配置文件

   ```shell
   cd ${VAF_DEPLOY_HOME_PATH}/runtime/html
   
   # vim cfg.py 修改服务端口等配置
   # vim ./start.sh  修改启动数量，取决于显卡数量
   
   sudo bash ./start.sh
   # 运行后新建 html_1 html_2 ... html_n 容器
   
   
   ########################################
   ################ html_n ################
   ########################################
   # 进入容器，并启动
   sudo docker exec -it html_1 bash
   # 进去容器后启动服务
   容器内> ./start.sh
   # 启动过程需输入密码，输入完成后ctrl+p+q退出前台
   # 重复上述过程，启动所有html_n，启动成功与否通过查看显存确认
   ```
   
   

## 10.安装启动自动测试镜像

```shell
# check md5
cd ${VAF_DEPLOY_HOME_PATH}/build/auto_test_deploy
bash ./check.sh
#合并镜像
cat auto_test_* > auto_testv2.0.tar
#解压镜像
sudo docker load < auto_testv2.0.tar
```
修改相应配置文件

```shell
cd ${VAF_DEPLOY_HOME_PATH}/build/auto_test_deploy/auto_test
#修改相关配置	TEST_FILE_URL请求地址 及WORKERS请求进程数
vim config.py
#修改映射数据存放地址 ，将第28行 /datasets_hdd/datasets/auto_test 替换成存放数据的目录
vim docker_start.sh
```
启动
```shell
bash ./docker_start.sh
#启动后进入同级logs目录查看progress_bars_main.txt文件是否正常输出日志
```

  
