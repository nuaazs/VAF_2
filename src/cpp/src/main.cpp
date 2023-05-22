#include <iostream>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "../include/timer.h"
#include "../include/search_best.h"

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

extern "C" {
    float test(DType vectorA[], int voicenum, int featsize, const char* shmid_file)
    {
        // 从shmid_file中读取共享内存的id shmid
        int shmid;
        FILE *fp = fopen(shmid_file, "r");
        if (fp == NULL)
        {
            std::cout << "Error: failed to open " << shmid_file << "!" << std::endl;
            return -1;
        }
        fscanf(fp, "%d", &shmid);
        fclose(fp);

        DType *pDB = (DType *)shmat(shmid, NULL, 0);

        Result res = SearchBest(static_cast<DType *>(vectorA), featsize, pDB, voicenum * featsize);
        float best_similarity = abs(res.similarity);
        int best_index = res.id;
        float result = best_similarity + best_index;
        if (res.similarity < 0)
        {
            result = -result;
        }
        return result;
    }
}

int main()
{
    const int VOICENUM = 88275;
    const int FEATSIZE = 192;
    const char* SHMID_FILE = "shmid.txt";

    DType *vectorA = new DType[FEATSIZE];
    for (int i = 0; i < FEATSIZE; i++)
    {
        vectorA[i] = rand() % 100;
    }

    float result = test(vectorA, VOICENUM, FEATSIZE, SHMID_FILE);
    std::cout << "result: " << result << std::endl;
}