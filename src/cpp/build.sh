#!/bin/bash
rm libexample.so
rm test
g++ main.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o test
g++ main.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -shared -fPIC -o libexample.so
# time ./test
