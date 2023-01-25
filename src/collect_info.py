import os
import pickle

from tqdm import tqdm


class CollectInformation:
    """
    Class is used to collect all required information about the dataset. It is helpful since we analyze data without
    loading them
    """

    def __init__(self, config_parameters: dict, dataset_type: str):
        """
        Method is used as initializer of the class
        :param config_parameters: Configuration parameters of the project
        :param dataset_type: type of dataset which can either be training or test dataset
        """
        self.configuration = self.set_configuration(config_parameters, dataset_type)
        self.info_data = self.process_for_models()

    @staticmethod
    def check_dir(directory: str) -> str:
        """
        Method is used to check the existence of provided directory. If it does not exist it creates the directory
        :param directory: provided path
        :return: the same path after checking it
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def set_configuration(self, parameters: dict, dataset_type: str) -> dict:
        """
        Method is used to collect task specific parameters from the provided project relevant parameters
        :param parameters: Configuration parameters of the project
        :param dataset_type: string that specifies dataset type
        :return: dictionary which includes task-specific parameters
        """

        input_path = os.path.join(parameters['dataset_dir'], f'data/{dataset_type}')
        information_dir = os.path.join(parameters['dataset_dir'], f'info_dir')
        information_dir = self.check_dir(information_dir)
        return {
            'data_type': dataset_type,
            'input_path': input_path,
            'info_dir': information_dir
        }

    def collect_pneumonia_info(self, data_type: str) -> list:
        """
        Method is used for collecting required information for pneumonia images according to the given data type
        :param data_type: specifies which kind of pneumonia images are analyzed
        :return: list in which each data is dictionary of image index and its path
        """
        check_name = '_bacteria' if data_type == 'bacteria' else '_virus'
        idx_info = 'b' if data_type == 'bacteria' else 'v'
        input_folder = os.path.join(self.configuration['input_path'], 'PNEUMONIA')
        ti = tqdm(iterable=enumerate(os.listdir(input_folder)), desc=f'Raw {data_type} data is collected')
        pneumonia_type = [{'idx': f'{idx_info}_{idx}', 'info': os.path.join(input_folder, img_name)} for idx, img_name
                          in ti if check_name in img_name]
        return pneumonia_type

    def collect_healthy_info(self) -> list:
        """
         Method is used for collecting required information for healthy images
         :return: list in which each data is dictionary of image index and its path
         """
        input_folder = os.path.join(self.configuration['input_path'], 'NORMAL')
        ti = tqdm(iterable=enumerate(os.listdir(input_folder)), desc=f'Raw healthy data is collected')
        healthy = [{'idx': f'h_{idx}', 'info': os.path.join(input_folder, img_name)} for idx, img_name in ti]
        return healthy

    def create_info_data(self) -> dict:
        """
        Method is used as collector, which collects all required raw info and combines them
        :return: dictionary in which each class for all models are specifically described
        """
        bacteria_info = self.collect_pneumonia_info('bacteria')
        virus_info = self.collect_pneumonia_info('virus')
        healthy_info = self.collect_healthy_info()
        pneumonia_info = bacteria_info + virus_info
        info_dict = {
            'b_class': {'info': bacteria_info, 'num_data': len(bacteria_info)},
            'v_class': {'info': virus_info, 'num_data': len(virus_info)},
            'h_class': {'info': healthy_info, 'num_data': len(healthy_info)},
            'p_class': {'info': pneumonia_info, 'num_data': len(pneumonia_info)}
        }

        return info_dict

    def process_for_models(self) -> dict:
        """
        Method is used as main function of the class which collects all required information according to each model
        :return: dictionary in which model names are keys, and values are relevant information per model
        """
        file_name = os.path.join(self.configuration['info_dir'], f"{self.configuration['data_type']}_info.pickle")
        if not os.path.exists(file_name):
            info_dict = self.create_info_data()
            model_classes = {
                'om': ['h_class', 'b_class', 'v_class'],
                'vb': ['b_class', 'v_class'],
                'ph': ['h_class', 'p_class']
            }
            model_dict = dict()
            for model_name, labels in model_classes.items():
                model_dict[model_name] = {each_label: info_dict[each_label] for each_label in labels}
            with open(file_name, 'wb') as info_file:
                pickle.dump(model_dict, info_file)

        with open(file_name, 'rb') as info_file:
            model_dict = pickle.load(info_file)
        return model_dict

    def __getitem__(self, model_name: str) -> dict:
        """
        Method is used as picker to pick model information according to the provided model name
        :param model_name: sting object to specify model name
        :return: dictionary for specified model
        """
        return self.info_data[model_name]
