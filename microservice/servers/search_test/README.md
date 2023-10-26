### 声纹检索模块
.
├── config.py                                           # 配置文件
├── demo.py
├── docker_start.sh
├── get_hotwords.py
├── log
│   ├── files_original-2023-10-24-104849.txt            # 原始文件,请求前会将所有需要处理文件路径放到这里
│   ├── processed_list.txt                              # 已经处理过的文件路径
│   └── run_2023-10-24_10-48.log                        # 运行时日志文件
├── nohup.out
├── rclone_sync.sh                                      # 同步本地文件到minio   用于上传url的请求方式
├── README.md
├── requirements.txt
├── run.py                                              # 启动文件                            
├── start.sh
├── start_test.sh
├── test.py
└── test.sh



```shell    
# 查看处理进度
tail -200f nohup.out |grep %

# 查看错误日志
cat nohup.out |grep "error"

# 统计VAD时长不够的数量
cat nohup.out |grep "VAD length is less than 10s" |wc -l

# 统计没有命中的数量
cat nohup.out |grep "is not in black list"|wc -l
```