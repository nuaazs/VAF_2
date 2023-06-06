#include <iostream>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
//main.cpp
#include <iostream>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../include/timer.h"
#include "../include/search_best.h"

#include <assert.h>
#include <cmath>
#include <float.h>
#include <climits>
#include <cblas.h>
#include "../include/cosine_similarity.h"

#define ALGIN                (32) // 使用SIMD需要内存对齐，128bit的指令需要16位对齐，256bit的指令需要32位对齐

typedef float DType;

using namespace std;

int main(int argc, char* argv[])
{
    if(argc < 3) {
        std::cout << "Usage: " << argv[0] << " FEATSIZE VECTOR_TXT_PATH SHMID_SAVE_PATH\n";
        return -1;
    }
    
    
    int FEATSIZE = atoi(argv[1]);
    char* vecFilename_txt = argv[2];
    // vecFilename = vecFilename_txt.replace(".txt", ".bin")
    char vecFilename[1024];
    strcpy(vecFilename, vecFilename_txt);
    vecFilename[strlen(vecFilename)-3] = 'b';
    vecFilename[strlen(vecFilename)-2] = 'i';
    vecFilename[strlen(vecFilename)-1] = 'n';
    cout << "vecFilename: " << vecFilename << endl;
    char* shmidFilename = argv[3];

    // VECTOR_TXT_PATH 行数为 VOICENUM
    // 读取VECTOR_TXT_PATH 行数

    std::ifstream infile(vecFilename_txt);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file: " << vecFilename_txt << std::endl;
        return 1;
    }
    // 统计行数
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    infile.close();
    int VOICENUM = lines.size(); // 文件行数即为 VOICENUM
    std::cout << "Number of lines in " << vecFilename_txt << ": " << VOICENUM << std::endl;

    DType* pDB = reinterpret_cast<DType*>(memalign(ALGIN, sizeof(DType)*VOICENUM*FEATSIZE));
    if(!pDB) {
        std::cout << "out of memory\n";
        return -1;
    }

    // 从文件中读取声纹底库特征  'vectorDB.bin' 是用python写的，用numpy保存的二进制文件, 是一个shape为(16948800,)的一维数组
    // 将vectorDB.bin读取结果保存在pDB中
    FILE* fp = fopen(vecFilename, "rb");
    if(!fp) {
        std::cout << "open file " << vecFilename << " failed.\n";
        return -1;
    }
    fread(pDB, sizeof(DType), VOICENUM*FEATSIZE, fp);
    fclose(fp);

    // T similarity = Cosine_similarity(pVecA, pVecDB + i*featsize, featsize);
    // 循环pDB中的每一个特征向量，计算当前声纹特征和剩余所有声纹特征的余弦相似度，获取大于0.8以上的相似度写入到txt中。
    // 余弦相似度计算公式：similarity = (A*B) / (|A|*|B|) 。
    // 两个向量的相似度计算可以调用函数 T similarity = Cosine_similarity(a,b featsize); 其中a为向量1，b为向量2，featsize为特征维数。
    // 结果txt格式为<id1>,<id2>,<similarity>一行一个

    for(int i = 0; i < VOICENUM; i++) {
        DType* pVecA = pDB + i*FEATSIZE;
        for(int j = i+1; j < VOICENUM; j++) {
            DType* pVecB = pDB + j*FEATSIZE;
            DType similarity = Cosine_similarity(pVecA, pVecB, FEATSIZE);
            if(similarity > 0.9) {
                //get the id of pVecA and pVecB
                string ida = lines[i];
                string idb = lines[j];
                std::cout << ida << "," << idb << "," << similarity << std::endl;
            }
        }
    }

    
    return 0;
}
