from pprint import pformat

from experiment import Experiment
from logger import Logger
from parameters_parser import parse_args

logger = Logger().logger(__name__)


def main():
    logger.info('Starting experiment')
    config = parse_args()
    logger.info(pformat(config))
    experiment = Experiment(config)
    experiment.run()
    Logger.shutdown(experiment.experiment_path)


if __name__ == '__main__':
    main()
