#!/bin/bash
# find all wav files in "/datasets_hdd/datasets/cjsd_download" and  "/datasets_hdd/cj_downloadwavs"
# and write them into train_wav.scp

find /datasets_hdd/datasets/cjsd_download -name "*.wav" > train_wav.scp
find /datasets_hdd/cj_downloadwavs -name "*.wav" >> train_wav.scp
