import torch
from pynvml import *

GIGABYTE = 1024 ** 3
def get_gpu_free_memory():
    try:
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        memory_free_values = []
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            memory_free_values.append(info.free / GIGABYTE)
        nvmlShutdown()
    except NVMLError as error:
        print("Failed to get GPU information:", error)
        memory_free_values = [-1]
    return memory_free_values


def get_unoccupied_device():
        memory_free_values = get_gpu_free_memory()
        if memory_free_values == [-1]:
            return torch.device('cpu')
        return torch.device('cuda:' + str(memory_free_values.index(max(memory_free_values))))


if __name__ == '__main__':
    device = get_unoccupied_device()
    print('Unoccupied Device:', device)
