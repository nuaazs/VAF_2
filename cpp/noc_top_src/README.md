## 功能
载入黑库，比对黑库

## 使用方法
```shell
# Build
bash build.sh

# 读取数据库至共享内存
../bin/read_db_multi_model --featsizes 256,256,256 --voicenum 27973 --bins /VAF/cpp/noc_top_src/test_data/101.bin,/VAF/cpp/noc_top_src/test_data/221.bin,/VAF/cpp/noc_top_src/test_data/293.bin

# 计算得分
../bin/noc_top1_multi_model_test -n 27973 -f 256,256,256 -t /VAF/cpp/noc_top_src/test_data/101.txt,/VAF/cpp/noc_top_src/test_data/221.txt,/VAF/cpp/noc_top_src/test_data/293.txt -d /VAF/cpp/noc_top_src/test_data/id.txt -i test.txt,test.txt,test.txt -k 12 -o test_out.txt
```
