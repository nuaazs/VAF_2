#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   calculate_handler.py
@Time    :   2023/10/30 17:11:30
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''
import os
from loguru import logger
from tools.cmd_handler import run_cmd
import cfg


def calculate_vaf(embedding_path_dict, output_path, topk=10):
    """
    calculate vaf
    Args:
        embedding_path_dict (dict): embedding path dict
        output_path (string): output path
        topk (int): topk
    Returns:
        list: topk list
    """
    memory_txt_path = "./cpp/noc_top_src/test_data"
    black_length = cfg.BLACK_SPEAKER_LENGTH
    feature_size = ""
    embedding_path_li = []
    for model in cfg.ENCODE_MODEL_LIST:
        feature_size += str(cfg.ENCODE_MODEL_FEATURE_DIM[model])+","
        embedding_path_li.append(embedding_path_dict[model.replace("_", "")])

    command = f"./cpp/bin/noc_top1_multi_model_test -n {black_length} -f {feature_size} -t {memory_txt_path}/101.txt,{memory_txt_path}/221.txt,{memory_txt_path}/293.txt -i {embedding_path_li[0]},{embedding_path_li[1]},{embedding_path_li[2]} -o {output_path} -d {memory_txt_path}/id.txt -k {topk}"
    logger.info(f"command: {command}")
    run_cmd(command)
    if os.path.exists(output_path):
        logger.info(f"calculate vaf success.")
        top_k = []
        with open(output_path, "r") as f:
            for line in f.readlines():
                top_k.append(line.strip().split(",")[1:])
        return top_k
    else:
        logger.error(f"calculate vaf failed.")
        return None


if __name__ == "__main__":
    calculate_vaf("/home/xuekaixiang/workplace/vaf/microservice/servers/vaf_server/cpp/noc_top_src/test.txt", "/home/xuekaixiang/workplace/vaf/microservice/servers/vaf_server/cpp/noc_top_src/test.txt",
                  "/home/xuekaixiang/workplace/vaf/microservice/servers/vaf_server/cpp/noc_top_src/test.txt", "./output.txt", topk=10)
