from src.precision import Precision
from src.runner import RunModelManager
from src.statistics import Statistics
from utils import *


def __main__() -> None:
    """
    Main function which process all steps of the project
    :return: None
    """
    parameters = get_parameters()

    for each_model in ['om', 'vb', 'ph']:
        stats = Statistics(parameters, each_model)
        stats.provide_statistics(True)
        print(f'{each_model} is prepared to train ...')
        if parameters[f'{each_model}_train']:
            runner = RunModelManager(parameters, each_model)

            print(runner.configuration['environment'])
            runner.train_model(True)

            stats.provide_statistics(False)

    precise = Precision(parameters)
    f1_score = precise.infer_results()
    print(f1_score)


if __name__ == '__main__':
    __main__()
