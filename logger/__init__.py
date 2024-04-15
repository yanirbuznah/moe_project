# logger_config.py
import logging
import os
import random
import shutil
import string

from utils.singleton_meta import SingletonMeta

# get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Logger(metaclass=SingletonMeta):
    def __init__(self, experiment_dir_path: str = None):
        experiment_dir_path = experiment_dir_path or os.path.join(PROJECT_ROOT, 'temp_logs', self._get_random_string())
        if not os.path.exists(experiment_dir_path):
            os.makedirs(experiment_dir_path)
        self._logger_file_name = os.path.join(experiment_dir_path, 'experiment.log')
        print(f"Logs will be saved in {self._logger_file_name}")

    def logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.logger_file_name)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger

    @staticmethod
    def _get_random_string(length=10):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for i in range(length))
        return random_string

    @property
    def logger_file_name(self):
        return self._logger_file_name

    @staticmethod
    def shutdown(copy_to=None):
        logger_file_name = Logger().logger_file_name
        logging.shutdown()
        if copy_to:
            shutil.move(logger_file_name, copy_to)
            print(f"Logs are saved in {copy_to}")

# Path: logger\logger.py
