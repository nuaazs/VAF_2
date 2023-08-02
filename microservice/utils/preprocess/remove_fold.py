import os
from pathlib import Path
from utils.log import logger
from utils.cmd import run_cmd
import cfg
def remove_fold_and_file(spkid):

    receive_path = f"{cfg.TEMP_PATH}"
    spk_dir = os.path.join(receive_path, str(spkid))
    run_cmd(f"rm -rf {spk_dir}")
