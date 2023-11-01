## noc_top_src

电信比对相关代码，功能：

sudo apt-get install libopenblas-dev

1. 读入声纹黑库bin，写入共享内存，对应可执行文件：./bin/read_db_multi_model

   ```
   ~/workplace/vaf/microservice/cpp/noc_top_src (microservice*) » ../bin/read_db_multi_model --voicenum 27973 --featsizes 256,256,256 --bins ./test_data/101.bin,test_data/221.bin,test_data/293.bin
   VoiceNum: 27973
   FeatSize: 256
   Need Memory: 28644352
   Shared memory ID for ./test_data/101.bin: 2
   VoiceNum: 27973
   FeatSize: 256
   Need Memory: 28644352
   Shared memory ID for test_data/221.bin: 3
   VoiceNum: 27973
   FeatSize: 256
   Need Memory: 28644352
   Shared memory ID for test_data/293.bin: 4
   
   
   会读取传入的几个bin，分别写入到共享内存，然后将共享内存的id写入到 <bin_path>.replace('.bin','.txt')
   ```

   

2. 读取声纹特征（多模型），和共享内存中的模型进行比对，对应可执行文件：./bin/noc_top1_multi_model_test

   ```
   输入数据待测声纹：
   	首先通过前面的步骤将多个模型的声纹特征提取后保存至多个txt中，比如分别保存在xuekaixiang_101.txt,xuekaixiang_221.txt,xuekaixiang_293.txt
   每个txt一行一个数，float
   
   ../bin/noc_top1_multi_model_test -n 27973 -f 256,256,256 -t ./test_data/101.txt,test_data/221.txt,test_data/293.txt -i ./test.txt,test.txt,test.txt -o xuekaixiang_out.txt -d test_data/id.txt -k 20
   
   
   ```

   