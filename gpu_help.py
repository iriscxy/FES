import csv
import os
import select
import socket
import subprocess
import sys
import torch


def get_free_gpu():
    child = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,utilization.memory', '--format=csv,noheader'],
        stdout=subprocess.PIPE)
    text = str(child.communicate()[0], 'utf8')
    reader = csv.reader(text.split('\n')[:-1])
    stats = []
    for line in reader:
        memory = int(line[1].strip()[:-4])  # MB
        compute_percentage = int(line[2].strip()[:-1])  # %
        memory_percentage = int(line[3].strip()[:-1])  # %
        stats.append({'id': line[0], 'memory': memory})
        if memory < 200 and compute_percentage < 5 and memory_percentage < 5:
            if line[0] == '3':# and get_host_ip() == '172.31.32.107':
                continue
            return int(line[0])
    stats.sort(key=lambda e: e['memory'])
    return int(stats[0]['id'])


def get_available_gpu():
    child = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,utilization.memory', '--format=csv,noheader'],
        stdout=subprocess.PIPE)
    text = str(child.communicate()[0], 'utf8')
    reader = csv.reader(text.split('\n')[:-1])
    stats = []
    for line in reader:
        memory = int(line[1].strip()[:-4])  # MB
        compute_percentage = int(line[2].strip()[:-1])  # %
        memory_percentage = int(line[3].strip()[:-1])  # %
        print(
            'index:{} memory:{} compute_percentage:{} memory_percentage:{}'.format(line[0], memory, compute_percentage,
                                                                                   memory_percentage))
        stats.append({'id': line[0], 'memory': memory})
        if memory < 200 and compute_percentage < 5 and memory_percentage < 5:
            if line[0] == '3' and get_host_ip() == '172.31.32.107':
                continue
            print('\033[32m gpu ' + line[0] + ' is available \033[0m')
            return int(line[0])
    stats.sort(key=lambda e: e['memory'])
    print('\033[31m can\'t find an available gpu, please change another server!!!! \033[0m')
    return int(get_input_with_timeout('use which GPU?', 10, stats[0]['id']))


def get_input_with_timeout(prompt: str, time: int = 10, default_input=None):
    """

    :param prompt: input prompt
    :param time: timeout in seconds
    :param default_input: when timeout this function will return this value
    :return: input value or default_input
    """
    print(prompt)
    print('Time Limit %d seconds' % time)
    i, o, e = select.select([sys.stdin], [], [], time)
    if i:
        return sys.stdin.readline().strip()
    else:
        return default_input


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def check_mem(gpu_id: int):
    devices_info = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[gpu_id].split(',')
    return total, used


def occupy_mem(gpu_id: int, occupy_rate: float):
    total, used = check_mem(gpu_id)
    total = int(total)
    used = int(used)
    max_mem = int(total * occupy_rate)
    block_mem = max_mem - used
    x = torch.FloatTensor(256, 1024, block_mem).to(gpu_id)
    del x
    return block_mem

