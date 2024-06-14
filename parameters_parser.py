import argparse
import logging
import os

import yaml
from yamlinclude import YamlIncludeConstructor

logger = logging.getLogger(__name__)


def load_args(config_path):
    if config_path is None:
        raise ValueError('No config file specified')
    if not config_path.endswith('.yml'):
        config_path += '.yml'
    if not os.path.exists(config_path):
        if os.path.exists(os.path.join('configs', config_path)):
            config_path = os.path.join('configs', config_path)
        else:
            raise ValueError('Config file not found')
    with open(config_path, 'r') as stream:
        try:
            YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.dirname(config_path))
            args = yaml.load(stream, Loader=yaml.FullLoader)
            if 'base_config' in args:
                base_config = load_args(args['base_config'])
                base_config.update(args)
                args = base_config
        except yaml.YAMLError as exc:
            logger.error('Error loading config file: %s', exc)
            raise
    return args



def parse_args():
    parser = argparse.ArgumentParser(description='Experiment Configuration')
    parser.add_argument('config')
    parse_args = parser.parse_args()
    args = load_args(parse_args.config)
    return args

