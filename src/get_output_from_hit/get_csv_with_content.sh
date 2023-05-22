#!/bin/bash
rm -rf ./log/*
rm -rf ./csvs/*
# python get_hit_csv.py

for i in {0..19}
do
    python mutli_process_text.py --worker_index=${i} --total_workers=20 >./log/get_content_${i}.log &
done

# wait for all the jobs to finish
wait
# merge all the results
python merge_all_subcsv.py
rm -rf ./csvs/*
rm hit_data.csv
# python download_wavs.py
cat output.csv | wc -l