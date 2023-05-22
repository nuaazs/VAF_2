## 说明文档
### auto_test
此目录为声纹项目的请求客户端代码

~~~
.
├── config.py                               # 配置文件
├── docker_start.sh                         # docker启动脚本
├── logs                                    # 日志文件夹
│   ├── 2023-04-26_test.log
│   └── test_report.txt


├── main.py                                 # 主程序
├── requirements.txt
├── start.sh                                # 启动脚本
├── start_test.sh                           # 启动测试脚本
└── test.py                                 # 测试程序
~~~

启动test.py脚本，会自动读取config.py中的配置，然后根据配置进行测试，测试结果会保存在logs目录下的日志文件中


### 构建镜像
- ~~~
  sh build.sh
  #注意版本
  ~~~
### 启动容器
- ~~~
  sh docker_start.sh
  #注意版本
  ~~~
  
  