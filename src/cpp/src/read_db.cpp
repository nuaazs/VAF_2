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

    // 创建一个共享内存区域，大小为 pDB 的大小
    int shmid = shmget(IPC_PRIVATE, sizeof(DType)*VOICENUM*FEATSIZE, IPC_CREAT | 0666);

    if(shmid == -1)
    {
        cout << "Error: failed to create shared memory!" << endl;
        return -1;
    }

    // 映射共享内存到当前进程的虚拟地址空间中
    char* memptr = (char*)shmat(shmid, NULL, 0);

    if(memptr == (char*)-1)
    {
        cout << "Error: failed to attach shared memory to virtual address space!" << endl;
        return -1;
    }

    // 将数据pDB写入共享内存

    memcpy(memptr, pDB, sizeof(DType)*VOICENUM*FEATSIZE);

    // 分离共享内存
    if(shmdt(memptr) == -1)
    {
        cout << "Error: failed to detach shared memory from virtual address space!" << endl;
        return -1;
    }

    // 打印共享内存标识符
    cout << "Shared memory ID: " << shmid << endl;
    // 共享内存标识符 写入shmid.txt
    FILE* fp_shmid = fopen(shmidFilename, "w");
    fprintf(fp_shmid, "%d", shmid);
    return 0;
}
