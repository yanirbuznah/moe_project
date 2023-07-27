from experiment import Experiment
from parameters_parser import parse_args
from pprint import pprint

from utils.general_utils import get_experiment_path


def main():
    config = parse_args()
    pprint(config)
    experiment = Experiment(config)
    experiment.run()



if __name__ == '__main__':
    main()
