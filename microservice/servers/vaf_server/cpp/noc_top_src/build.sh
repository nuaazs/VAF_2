#!/bin/bash
g++ read_db_multi_model.cpp -std=c++17 -o ../bin/read_db_multi_model
g++ main.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o ../bin/noc_top1_multi_model_test
# g++ main.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -shared -fPIC -o ../bin/noc_top1_multi_model.so
