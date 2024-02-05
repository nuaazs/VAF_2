#include <iostream>
#include <cstring>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <vector>
#include <sstream>
#include <cstdlib>

#define ALGIN                (32)

typedef float DType;

using namespace std;

vector<string> split_string(const string &str, char delimiter)
{
    vector<string> result;
    stringstream ss(str);
    string token;
    while (getline(ss, token, delimiter))
    {
        result.push_back(token);
    }
    return result;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        cout << "usage: " << argv[0] << " --voicenum <VOICENUM> --featsizes <comma-separated featsizes> --bins <comma-separated bins>" << endl;
        return -1;
    }

    int VOICENUM = 0;
    vector<string> featsizes_str, bins_str;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--voicenum") == 0 && i+1 < argc) {
            VOICENUM = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "--featsizes") == 0 && i+1 < argc) {
            featsizes_str = split_string(string(argv[i+1]), ',');
        } else if (strcmp(argv[i], "--bins") == 0 && i+1 < argc) {
            bins_str = split_string(string(argv[i+1]), ',');
        }
    }

    if (featsizes_str.empty() || featsizes_str.size() != bins_str.size()) {
        cout << "Error: invalid --voicenum, --featsizes, or --bins arguments" << endl;
        return -1;
    }

    vector<int> featsizes(featsizes_str.size());
    for (int i = 0; i < featsizes_str.size(); i++) {
        featsizes[i] = atoi(featsizes_str[i].c_str());
    }

    int shmid;
    DType* pDB;

    for (int i = 0; i < bins_str.size(); i++) {
        string bin_filename = bins_str[i];
        string txt_filename = bin_filename.substr(0, bin_filename.size()-4) + ".txt";

        pDB = reinterpret_cast<DType*>(aligned_alloc(ALGIN, sizeof(DType)*VOICENUM*featsizes[i]));
        if(!pDB) {
            std::cout << "out of memory\n";
            return -1;
        }

        FILE* fp = fopen(bin_filename.c_str(), "rb");
        if(!fp) {
            std::cout << "open file " << bin_filename << " failed.\n";
            return -1;
        }
        fread(pDB, sizeof(DType), VOICENUM*featsizes[i], fp);
        fclose(fp);
        cout << "VoiceNum: " << VOICENUM << endl;
        cout << "FeatSize: " << featsizes[i] << endl;
        cout << "Need Memory: " << sizeof(DType)*VOICENUM*featsizes[i] << endl;
        shmid = shmget(IPC_PRIVATE, sizeof(DType)*VOICENUM*featsizes[i], IPC_CREAT | 0666);

        if(shmid == -1)
        {
            cout << "Error: failed to create shared memory!" << endl;
            return -1;
        }

        char* memptr = (char*)shmat(shmid, NULL, 0);

        if(memptr == (char*)-1)
        {
            cout << "Error: failed to attach shared memory to virtual address space!" << endl;
            return -1;
        }

        memcpy(memptr, pDB, sizeof(DType)*VOICENUM*featsizes[i]);

        if(shmdt(memptr) == -1)
        {
            cout << "Error: failed to detach shared memory from virtual address space!" << endl;
            return -1;
        }

        cout << "Shared memory ID for " << bin_filename << ": " << shmid << endl;
        FILE* fp_shmid = fopen(txt_filename.c_str(), "w");
        fprintf(fp_shmid, "%d", shmid);

        free(pDB);
    }

    return 0;
}
