import subprocess

import torch

from logger import Logger

logger = Logger().logger(__name__)


def get_unoccupied_device():
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        memory_free_info = _output_to_list(subprocess.check_output(command.split()))
        memory_free_values = [int(x) for i, x in enumerate(memory_free_info)]

        return torch.device('cuda:' + str(memory_free_values.index(max(memory_free_values))))
    except Exception as e:
        logger.warning('"nvidia-smi" is probably not installed. GPUs are not usable. Error:', e)
        return torch.device('cpu')


if __name__ == '__main__':
    device = get_unoccupied_device()
    print('Unoccupied Device:', device)
