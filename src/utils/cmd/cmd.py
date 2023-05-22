import subprocess
from utils.log import logger
def run_cmd(cmd):
    """run shell command.
    Args:
        cmd (string): shell command.
    Returns:
        string: result of shell command.
    """
    logger.info(f"Run command: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        logger.info(f"Command: {cmd}, result output: {result.stdout}")
        logger.error("run command error!")
        raise Exception("run command error!")
    else:
        logger.info(f"Command: {cmd}, result output: {result.stdout}")
    return result.stdout.decode('utf-8')

if __name__ == '__main__':
    run_cmd("ffmpeg -i /tmp/99926797/raw_1_4060068.wav -y -ss 0 -to 90 -ar 16000 -ac 1 -vn -map_channel 0.0.1 /tmp/99926797/raw_1_4074027.wav")