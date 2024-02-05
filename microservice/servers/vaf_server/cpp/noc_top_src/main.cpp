#include <iostream>
#include <fstream>  // 读写文件流
#include <malloc.h>
#include "timer.h"
#include "search_best.h"
#include <string>
#include <vector>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#define ALGIN (32)       // 使用SIMD需要内存对齐，128bit的指令需要16位对齐，256bit的指令需要32位对齐
#define FEATSIZE (256)   // 每个声纹特征向量的维度是192维，每一维是一个DType类型的浮点数
#define FLOAT_MIN (1e-10f)

using namespace std;
typedef float DType;

float calcL(const DType *const pVec, const int len)
{
    float l = 0.0f;

    for (int i = 0; i < len; i++)
    {
        l += pVec[i] * pVec[i];
    }

    return sqrt(l) + FLOAT_MIN;
}

extern "C" {
    float test(vector<const DType*>& vecVecA, int voicenum, vector<const DType*>& vecDB, int topk, string output_path, string id_txt)
    {
        // 打印输入参数
        // std::cout << "vecVecA.size() = " << vecVecA.size() << std::endl;
        std::cout << "voicenum = " << voicenum << std::endl;
        // 计算多个txt读取到的数据库结果的平均值的最大值
        std::vector<Result> res = SearchMeanBest(vecVecA, FEATSIZE, vecDB, voicenum * FEATSIZE, topk);
        
        // write to output file
        
        ofstream fout(output_path);
        if (!fout.is_open()) {
            cout << "Error: failed to open " << output_path << endl;
            return -1;
        }
        for (int i = 0; i < res.size(); i++) {
            float best_similarity = abs(res[i].similarity);
            int best_index = res[i].id;
            best_index += 1;
            //读取 id.txt 文件的best_index行，获取best_id
            ifstream fin(id_txt);
            if (!fin.is_open()) {
                cout << "Error: failed to open " << id_txt << endl;
                return -1;
            }
            string best_id;
            for (int j = 0; j < best_index; j++) {
                fin >> best_id;
            }
            fin.close();

            // float result sign equals to res.similarity sign
            float result = best_similarity;
            if (res[i].similarity < 0)
            {
                result = -result;
            }
            fout << best_index << "," << best_id << "," << result << std::endl;
        }
        fout.close();
        
        return 0.0f;
    }
}

int main(int argc, char* argv[])
{
    string input_paths;  // 用于存储输入文件路径，多个文件用逗号分隔
    string output_path;  // 用于存储输出文件路径
    string id_path;      // 用于存储id的txt文件路径
    int voicenum = 0;
    int topk = 0;
    vector<int> featsizes;
    vector<string> txts;
    int opt;
    while ((opt = getopt(argc, argv, "n:f:t:i:o:d:k:")) != -1) {
        switch(opt){
            case 'n':{
                voicenum = atoi(optarg);
                break;
            }
            case 'f':{
                char* p = strtok(optarg,",");
                while(p != NULL){
                    featsizes.push_back(atoi(p));
                    p = strtok(NULL,",");
                }
                break;
            }
            case 't':{
                char* p = strtok(optarg,",");
                while(p != NULL){
                    txts.push_back(p);
                    p = strtok(NULL,",");
                }
                break;
            }
            case 'i':{
                input_paths = optarg;
                break;
            }
            case 'o':{
                output_path = optarg;
                break;
            }
            case 'd':{
                id_path = optarg;
                break;
            }
            case 'k':{
                topk = atoi(optarg);
                break;
            }
            default:{
                cout<<"Invalid argument."<<endl;
                return -1;
            }
        }
    }

    if (input_paths.empty() || output_path.empty() || voicenum == 0 || featsizes.empty() || txts.empty() || id_path.empty() || topk == 0) {
        cout << "Usage: " << argv[0] << " -n <voicenum> -f <comma-separated featsizes> -t <comma-separated txts> -i <comma-separated input-txt-paths> -o <output-txt-path> -d <id-txt-path> -k <topk>" << endl;
        return -1;
    }

    // 处理输入文件路径，将多个路径分别存入vector中
    vector<string> input_paths_vec;
    char* p = strtok(&input_paths[0],",");
    while(p != NULL){
        input_paths_vec.push_back(p);
        p = strtok(NULL,",");
    }

    // 读取输入文件中的数据，存到vectorA数组中
    vector<const DType*> vecVecA;
    for (auto input_path : input_paths_vec) {
        DType *vectorA = new DType[featsizes[0]];
        ifstream fin(input_path);
        if (!fin.is_open()) {
            cout << "Error: failed to open " << input_path << endl;
            return -1;
        }
        for (int i = 0; i < featsizes[0]; ++i) {
            fin >> vectorA[i];
        }
        fin.close();
        vecVecA.push_back(vectorA);
    }

    vector<const DType*> vecDB;
    for (auto txt : txts) {
        // 从txt文件中读取共享内存的id shmid
        int shmid;
        FILE *fp = fopen(txt.c_str(), "r");
        if (fp == NULL)
        {
            cout << "Error: failed to open " << txt << endl;
            return -1;
        }
        fscanf(fp, "%d", &shmid);
        fclose(fp);

        DType *pDB = (DType *)shmat(shmid, NULL, 0);
        vecDB.push_back(pDB);
    }

    // 调用test函数
    test(vecVecA, voicenum, vecDB, topk, output_path, id_path);

    for (auto vecA : vecVecA) {
        delete[] vecA;
    }
    return 0;
}
