#!/bin/bash
rm libexample.so
rm test
g++ main.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o test
g++ n2_score.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o ../bin/n2_score
g++ main.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -shared -fPIC -o ../lib/get_top.so
g++ read_db.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o ../bin/read_db
# time ./test