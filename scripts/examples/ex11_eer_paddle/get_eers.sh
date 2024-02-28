#!/bin/bash
for i in {0..19}
do
    python get_eer_pipline.py --worker_idx=${i} >./log/get_eer_${i}.log &
done
