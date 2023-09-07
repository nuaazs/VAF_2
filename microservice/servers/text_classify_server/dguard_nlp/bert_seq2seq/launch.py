# Copyright 2020 The Microsoft DeepSpeed Team
"""
sailing runner is the main front-end to launching multi-worker
training jobs with DeepSpeed. By default this uses pdsh to parallel
ssh into multiple worker nodes and launch all the necessary processes
per rank for training.
"""

import os
import sys
import json
import subprocess
import collections
import socket
import signal
import logging

import torch.distributed as dist


def fetch_hostfile(hostfile_path):
    if not os.path.isfile(hostfile_path):
        print("Unable to find hostfile, will proceed with training "
              "with local resources only.")
        return None
    # e.g., worker-0 slots=16
    with open(hostfile_path, 'r') as fd:
        resource_pool = collections.OrderedDict()
        for line in fd.readlines():
            line = line.strip()
            if line == '':
                # skip empty lines
                continue
            try:
                hostname, slots = line.split()
                _, slot_count = slots.split("=")
                slot_count = int(slot_count)
            except ValueError as err:
                raise err
            if hostname in resource_pool:
                raise ValueError(f"host {hostname} is already defined")
            resource_pool[hostname] = slot_count

    return resource_pool


def cmd_load_hyperparam(config_path=None, format="json", encoding="utf-8"):
    """
    shell load arguments form argparse and config file
    """
    # config_path='config/config_block_large_chinese.json'
    format = config_path.rsplit('.')[-1]
    with open(config_path, 'r', encoding=encoding) as f:
        if format == "json":
            config_dict = json.load(f)
        else:
            raise NameError("current format%s for hyperparam file is invalid" %
                            format)
    config_cmd = []
    for key in config_dict:
        if len(str(config_dict[key])) == 0:
            config_cmd.append('--' + key)
        else:
            config_cmd.append('--' + key)
            config_cmd.append(str(config_dict[key]))
    return config_cmd


def launch_dist(
               env_type="DDP",
               num_nodes=1,
               gpus_per_node=1,
               master_addr='localhost',
               master_port=17500,
               training_script='train.py',
               hostfile=None,
               ):

    if num_nodes != 1:
        print("多机多卡待测试。暂不支持。")
        os._exit(0)
    if env_type == "DDP":
        cmd_launch = []
        cmd_launch.extend([
            # 'export NUM_NODES=' + str(num_nodes) + ';',
            # 'export GPUS_PER_NODE=' + str(gpus_per_node) + ';',
            sys.executable,
            # "python",
            '-m', 'torch.distributed.launch'
        ])
        torch_distributed_args = [
            '--nproc_per_node',
            str(gpus_per_node),
            '--nnodes',
            str(num_nodes),
            '--node_rank',
            str(0),
            '--master_addr',
            master_addr,
            '--master_port',
            str(master_port),
        ]
        cmd_launch.extend(torch_distributed_args)
        cmd_launch.append(training_script)
        cmd_launch.append('--not_call_launch')
        run_cmd = ' '.join(cmd_launch)
        p = subprocess.Popen(run_cmd, shell=True, preexec_fn=os.setsid)
        def signal_handler(signal, frame):
            os.killpg(os.getpgid(p.pid), 9)
        signal.signal(signal.SIGINT, signal_handler)
        p.wait()
        print ('finish')

    elif env_type == "deepspeed":

        # if hostfile is None:
        #     print(
        #         'Unable to find hostfile, will proceed with training with local resources only.'
        #     )
        #     os.makedirs("./tmp", exist_ok=True)
        #
        #     with open('./tmp/hostfile', 'w') as w:
        #         w.write(socket.gethostname() + ' slots=2')
        #     hostfile = './tmp/hostfile'

        cmd_launch = ['deepspeed']

        cmd_launch.extend([
            '--master_port',
            str(master_port),
            '--num_nodes',
            str(num_nodes),
            '--num_gpus',
            str(gpus_per_node),
            # '--hostfile',
            # hostfile,
        ])

        cmd_launch.append(training_script)

        cmd_launch.append('--not_call_launch')
        run_cmd = ' '.join(cmd_launch)
        p = subprocess.Popen(run_cmd, shell=True, preexec_fn=os.setsid)
        def signal_handler(signal, frame):
            os.killpg(os.getpgid(p.pid), 9)
        signal.signal(signal.SIGINT, signal_handler)
        p.wait()
        print ('finish')

    elif env_type == "deepspeed+mpu":

        # if hostfile is None:
        #     print(
        #         'Unable to find hostfile, will proceed with training with local resources only.'
        #     )
        #     os.makedirs("./tmp", exist_ok=True)
        #
        #     with open('./tmp/hostfile', 'w') as w:
        #         w.write(socket.gethostname() + ' slots=2')
        #     hostfile = './tmp/hostfile'

        cmd_launch = ["export ENV_TYPE=deepspeed+mpu;",'deepspeed']

        cmd_launch.extend([
            '--master_port',
            str(master_port),
            '--num_nodes',
            str(num_nodes),
            '--num_gpus',
            str(gpus_per_node),
            # '--hostfile',
            # hostfile,
        ])

        cmd_launch.append(training_script)

        cmd_launch.append('--not_call_launch')
        run_cmd = ' '.join(cmd_launch)
        p = subprocess.Popen(run_cmd, shell=True, preexec_fn=os.setsid)
        def signal_handler(signal, frame):
            os.killpg(os.getpgid(p.pid), 9)
        signal.signal(signal.SIGINT, signal_handler)
        p.wait()
        print ('finish')

    else :
        print("不支持的env_type")
        os._exit(0)
