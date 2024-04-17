import traceback
from pprint import pformat

from experiment import Experiment
from logger import Logger
from parameters_parser import parse_args
import wandb




def main():
    config = parse_args()
    wandb_project_name = config['log'].get('wandb_project_name', None)
    if wandb_project_name is not None:
        wandb.init(project=wandb_project_name, config=config)
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
        exit(0)
    logger.info('Experiment finished')

if __name__ == '__main__':
    main()
