#!/bin/bash
# rm -rf ./pretrained_models
# rm -rf ../../cache/eer_light/
# python get_embeddings_and_pairs.py

for i in {0..89}
do
    python get_eer_pipline.py --worker_idx=${i} --total_worker=90 >./log/get_eer_${i}.log &
done

# wait for all the jobs to finish
wait
# merge all the results
python get_all_eer.py
