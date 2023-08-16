import subprocess
from loguru import logger
import os
def run_cmd(cmd,check=True,util_exist=None):
    """run shell command.
    Args:
        cmd (string): shell command.
    Returns:
        string: result of shell command.
    """
    logger.info(f"Run command: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if util_exist:
        test_time = 0
        if ((not os.path.exists(util_exist)) or os.path.getsize(util_exist) < 1000) and test_time<10:
            test_time = test_time +1
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check:
        if result.returncode != 0:
            logger.info(f"Command: {cmd}, result output: {result.stdout}")
            logger.error("run command error!")
            raise Exception(f"run command error! \n cmd :{cmd} cmd result: {result.stdout}")
        else:
            logger.info(f"Command: {cmd}, result output: {result.stdout}")
    else:
        logger.info(f"Command: {cmd}, result output: {result.stdout}")
    return result.stdout.decode('utf-8')

def remove_father_path(filepath):
    father_path = os.path.dirname(filepath)
    run_cmd(f"rm -rf {father_path}")
if __name__ == '__main__':
    run_cmd("ffmpeg -i /tmp/99926797/raw_1_4060068.wav -y -ss 0 -to 90 -ar 16000 -ac 1 -vn -map_channel 0.0.1 /tmp/99926797/raw_1_4074027.wav")