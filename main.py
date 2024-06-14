import pickle
import traceback
from itertools import product
from pprint import pformat
import wandb

from logger import Logger #, init_logger
# if not Logger.initialized:
#     init_logger()
from experiment import Experiment
from parameters_parser import parse_args


def run_experiment(config):
    logger = Logger().logger(__name__)
    logger.info('Starting experiment')
    logger.info(pformat(config))
    experiment = Experiment(config)
    try:
        experiment.run()
    except Exception as e:
        logger.fatal(f'Exception occurred during experiment:\n {traceback.format_exc()}')
    finally:
        Logger.shutdown(experiment.experiment_path)
        wandb.finish()
    logger.info('Experiment finished')


def main():
    config = parse_args()
    wandb_project_name = config['log'].get('wandb_project_name', None)

    def set_value_by_key_from_nested_dict(nested_dict, key, value):
        for k, v in nested_dict.items():
            key_name = list(key.keys())[0] if isinstance(key, dict) else key
            if k == key_name:
                if isinstance(v, dict):
                    # next_key = list(key[k].keys())[0]
                    set_value_by_key_from_nested_dict(v, key[key_name], value)
                elif isinstance(v, list):
                    set_value_by_key_from_nested_dict(v[0], key[key_name][0], value)
                else:
                    nested_dict[k] = value



    if 'changes' in config:
        # Generate all combinations of options for each change
        options_product = product(*(change['options'] for change in config['changes']))

        # Iterate over each combination
        for options in options_product:
            # Set the values for each key-option pair
            for change, option in zip(config['changes'], options):
                set_value_by_key_from_nested_dict(config, change['key'], option)

            # Initialize wandb if project name is provided
            if wandb_project_name is not None:
                wandb.init(project=wandb_project_name, config=config)
            #
            # Run the experiment with the current configuration
            run_experiment(config)

    else:
        if wandb_project_name is not None:
            wandb.init(project=wandb_project_name, config=config)
        run_experiment(config)


if __name__ == '__main__':
    main()
