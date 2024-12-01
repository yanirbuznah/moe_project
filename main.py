#!/usr/bin/env python3

import traceback
from datetime import datetime
from itertools import product
from pprint import pformat

import numpy as np
# Load model directly
import pandas as pd

import wandb
# if not Logger.initialized:
#     init_logger()
from experiment import Experiment
from logger import Logger  # , init_logger
from parameters_parser import parse_args


def run_experiment(config, run=None):
    logger = Logger().logger(__name__)
    logger.info('Starting experiment')
    logger.info(pformat(config))
    try:
        experiment = Experiment(config)
        experiment.run()
    except Exception as e:
        logger.fatal(f'Exception occurred during experiment:\n {traceback.format_exc()}')
    finally:
        if run:
            df = pd.read_csv('results/runs_summary_base1.csv')
            df = df.append({'id': run.id, 'start_time': datetime.fromtimestamp(run.start_time), 'dir': run.dir,
                            'link': run.get_url(), 'criterion': str(experiment.model.criterion),
                            'project_name': config['log'].get('wandb_project_name', None),
                            'experiment_name': config['log']['experiment_name'],
                            'dataset': config['dataloader']['dataset'],
                            'accuracy': run.summary.get('max.validate.Accuracy', 0),
                            'min_loss': run.summary.get('min.validate.Loss', -1),
                            'comments': ''},
                           ignore_index=True)
            df.to_csv('results/runs_summary_base1.csv', index=False)
        acc = run.summary.get('max.validate.Accuracy', 0)
        Logger.shutdown(experiment.experiment_path)
        wandb.finish()
        logger.info('Experiment finished')
        return acc


def main():
    config = parse_args()
    wandb_project_name = config['log'].get('wandb_project_name', None)
    run = None
    experiments_acc = []

    def set_value_by_key_from_nested_dict(nested_dict, key, value):
        key_name = list(key.keys())[0] if isinstance(key, dict) else key
        for k, v in nested_dict.items():
            if k == key_name:
                if isinstance(v, dict):
                    set_value_by_key_from_nested_dict(v, key[key_name], value)
                else:
                    nested_dict[k] = value

    if 'changes' in config:
        # change the order so it will
        config['changes'].reverse()
        # Generate all combinations of options for each change
        options_product = product(*(change['options'] for change in config['changes']))

        # Iterate over each combination
        for options in options_product:
            # Set the values for each key-option pair
            for change, option in zip(config['changes'], options):
                set_value_by_key_from_nested_dict(config, change['key'], option)

            # Initialize wandb if project name is provided
            if wandb_project_name is not None:
                # wandb.debug = True
                run = wandb.init(project=wandb_project_name, config=config)
                run.log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".yml"))

            #
            # Run the experiment with the current configuration
            acc = run_experiment(config, run)
            print(f"Accuracy: {acc}")
            experiments_acc.append(acc)

            # print(config)
        return experiments_acc
    else:
        if wandb_project_name is not None:
            run = wandb.init(project=wandb_project_name, config=config)
        return [run_experiment(config, run)]


if __name__ == '__main__':
    accs = [0.9197102575488455, 0.9181005772646537, 0.9203485790408525, 0.9184336145648313, 0.9194604795737122,
            0.9191274422735346]
    for i in range(4):
        accs.extend(main())
        print(accs)
    print(np.mean(accs))
    print(np.std(accs))
