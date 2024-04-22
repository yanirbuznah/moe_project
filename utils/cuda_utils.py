import subprocess

import torch

from logger import Logger

import torch
from pynvml import *

logger = Logger().logger(__name__)

def get_unoccupied_device():
    try:
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        memory_free_values = []
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            memory_free_values.append(info.free)
        nvmlShutdown()

        return torch.device('cuda:' + str(memory_free_values.index(max(memory_free_values))))
    except NVMLError as error:
        print("Failed to get GPU information:", error)
        return torch.device('cpu')



if __name__ == '__main__':
    device = get_unoccupied_device()
    print('Unoccupied Device:', device)
