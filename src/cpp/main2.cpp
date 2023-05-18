// # coding = utf-8
// # @Time    : 2023-04-25  15:22:35
// # @Author  : zhaosheng@nuaa.edu.cn
// # @Describe: .

#include <iostream>
#include <malloc.h>
#include "timer.h"
#include "search_best.h"
#include <string>
#define ALGIN (32)       // 使用SIMD需要内存对齐，128bit的指令需要16位对齐，256bit的指令需要32位对齐
#define VOICENUM (88275) // 底库中存有88275声纹特征向量
#define FEATSIZE (192)   // 每个声纹特征向量的维度是192维，每一维是一个DType类型的浮点数
#include <iostream>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>

using namespace std;
typedef float DType;




float calcL(const DType *const pVec, const int len)
{
    float l = 0.0f;

    for (int i = 0; i < len; i++)
    {
        l += pVec[i] * pVec[i];
    }

    return sqrt(l) + FLT_MIN;
}

// std::string test(DType vectorA[]);
extern "C" {
    float test(DType vectorA[],DType pDB[])
    {
        // int shmid = 4;
        // DType *pDB = (DType *)shmat(shmid, NULL, 0);

        Result res = SearchBest(static_cast<DType *>(vectorA), FEATSIZE, pDB, VOICENUM * FEATSIZE);
        // best_similarity equals to abs of res.similarity
        float best_similarity = abs(res.similarity);
        int best_index = res.id;
        // return res;
        float result = best_similarity + best_index;
        // float result sign equals to res.similarity sign
        if (res.similarity < 0)
        {
            result = -result;
        }
        return result;
    }
}

int main()
{return 0;}