#!/bin/bash

# INSTALLA DEPENDENCIES
pip install tqdm npy
apt-get install csvtool

# CALC
python data4_test.py --register_npy_path xxx --test_npy_path xxx --result 3_result.csv
python data4_test.py --register_npy_path xxx --test_npy_path xxx --result 5_result.csv
python data4_test.py --register_npy_path xxx --test_npy_path xxx --result 8_result.csv
python data4_test.py --register_npy_path xxx --test_npy_path xxx --result 10_result.csv
python data4_test.py --register_npy_path xxx --test_npy_path xxx --result 12_result.csv

# TABLE PRINT RESULTs
csvtool readable 3_result.csv | view -
csvtool readable 5_result.csv | view -
csvtool readable 8_result.csv | view -
csvtool readable 10_result.csv | view -
csvtool readable 12_result.csv | view -
