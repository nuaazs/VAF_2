#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cmd_handler.py
@Time    :   2023/10/14 16:54:59
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''

from loguru import logger
import subprocess


def run_cmd(cmd):
    """
    Run shell command.
    Args:
        cmd (string): shell command.
    Returns:
    """
    max_retries = 10
    retries = 0
    success = False
    logger.info(f"Run command: {cmd}")
    while not success and retries < max_retries:
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True)
            success = True
        except subprocess.CalledProcessError as e:
            logger.error(f"发生了CalledProcessError错误: {str(e)}. ")
            retries += 1
        except Exception as e:
            logger.error(f"发生了其他错误: {str(e)}. ")
    if not success:
        logger.error(f"达到最大重试次数 {max_retries}，无法执行ffmpeg命令。")
