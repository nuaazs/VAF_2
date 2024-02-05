#!/bin/bash
rm read_db
g++ read_db.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o read_db
./read_db

