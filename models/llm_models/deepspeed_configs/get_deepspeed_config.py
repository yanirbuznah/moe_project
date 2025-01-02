import json
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def get_deepspeed_config(config:str):
    if config.lower() == 'zero1':
        config = json.load(open(f'{BASE_PATH}/ZeRO-1-config.json', 'r'))
    elif config.lower() == 'zero2':
        config = json.load(open(f'{BASE_PATH}/ZeRO-2-config.json', 'r'))
    elif config.lower() == 'zero3':
        config = json.load(open(f'{BASE_PATH}/ZeRO-3-config.json', 'r'))
    else:
        raise FileNotFoundError('Config not found')
    return config


def main():
    config1 = get_deepspeed_config('zero1')
    config2 = get_deepspeed_config('zero2')
    config3 = get_deepspeed_config('zero3')
    print(config1)
    print(config2)
    print(config3)

if __name__ == '__main__':
    main()