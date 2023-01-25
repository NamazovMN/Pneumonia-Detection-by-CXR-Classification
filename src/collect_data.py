import os.path
import pickle

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
from tqdm import tqdm

from src.collect_info import CollectInformation


class CollectDataset:
    def __init__(self, config_parameters: dict, dataset_type: str, model_type: str, zero: bool = True):
        """
        Method is used as initializer of the class
        :param config_parameters: Configuration parameters of the project
        :param dataset_type: type of dataset which can either be training or test dataset
        :param model_type: name of the model
        :param zero: specifies whether zero centering will be done or not
        """
        self.configuration = self.set_configuration(config_parameters, dataset_type, model_type)
        self.dataset = self.collect_dataset(zero=zero)
        self.labels = self.get_labels()

    def get_labels(self) -> dict:
        """
        Method is used for collecting labels
        :return: dictionary in which keys are labels and values are their indexes
        """
        labels_dict_path = os.path.join(self.configuration['data_env'], 'labels_dict.pickle')
        if not os.path.exists(labels_dict_path):
            model_data_info = self.configuration['info_object'][self.configuration['model_name']]
            labels_dict = {label: idx for idx, label in enumerate(model_data_info.keys())}
            with open(labels_dict_path, 'wb') as labels_data:
                pickle.dump(labels_dict, labels_data)
        with open(labels_dict_path, 'rb') as labels_data:
            labels_dict = pickle.load(labels_data)
        return labels_dict

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

    def set_configuration(self, parameters: dict, dataset_type: str, model_type: str) -> dict:
        """
        Method is used to collect task specific parameters from the provided project relevant parameters
        :param parameters: Configuration parameters of the project
        :param dataset_type: string that specifies dataset type
        :param model_type: name of the model
        :return: dictionary which includes task-specific parameters
        """
        input_data = os.path.join(parameters['dataset_dir'], 'data')
        output_dir = os.path.join(parameters['dataset_dir'], f'processed_data/{model_type}/'
                                                             f'{"uniform" if parameters["uniform"] else "raw"}')
        output_dir = self.check_dir(output_dir)
        out_file = os.path.join(output_dir, f'{dataset_type}.pickle')
        zero_out_file = os.path.join(output_dir, f'{dataset_type}_zero_center.pickle')
        return {
            'zero_center': parameters['zero_center'],
            'data_env': output_dir,
            'info_object': CollectInformation(parameters, dataset_type),
            'uniform': parameters['uniform'],
            'desired_dim': parameters['image_dim'],
            'input_dir': input_data,
            'model_name': model_type,
            'output_dir': out_file,
            'zero_out_dir': zero_out_file,
            'dataset_type': dataset_type
        }

    @staticmethod
    def compute_combination(original_num: int, max_label: int) -> list:
        """
        Method is used to compute how many augmentation for how much data will be done. Each scenario is added to the
        list as a result. In case original num is 2 times smaller than the number of data for max label, then at least
        2 augmentations will be done.
        :param original_num: original number of data for specific label
        :param max_label: number of data for the label with maximum number
        :return: list in which index specifies augmentation type and value specifies number of data that augmentation
                 will be applied
        """

        differences = list()
        if max_label - original_num:
            difference = max_label - original_num if max_label - original_num < original_num else original_num
            for count in range(original_num, max_label, difference):
                if count + difference < max_label:
                    differences.append(difference)
                else:
                    differences.append(max_label - count)
        else:
            differences.append(0)
        return differences

    def compute_uniform_requirements(self, model_data_info: dict) -> dict:
        """
        Method is used to check which label has the highest number of data and to compute differences between that
        number and number of data for other labels
        :param model_data_info:
        :return: dictionary in which keys are labels, values are differences between labels wrt max label in terms of
                 data amount
        """
        num_data_per_label = {each_label: model_info['num_data'] for each_label, model_info in model_data_info.items()}
        max_label = max(num_data_per_label, key=num_data_per_label.get)
        transformation_info = dict()
        for label, original_num in num_data_per_label.items():
            original_num = num_data_per_label[label]
            transformation_info[label] = self.compute_combination(original_num, num_data_per_label[max_label])
        return transformation_info

    def collect_dataset(self, zero: bool = False) -> list:
        """
        Method is used to collect dataset. If zero is True, then zero centered data will be sent, otherwise original
        version of data will be sent
        :param zero: boolean variable specifies if zero centering will be utilized (True) or not (False)
        :return: list of elements, in which each element is dictionary of image idx, image and its label
        """
        if not os.path.exists(self.configuration['output_dir']):
            model_data_info = self.configuration['info_object'][self.configuration['model_name']]
            dataset_dict = list()
            transformation = self.compute_uniform_requirements(model_data_info)

            for label, data in model_data_info.items():
                ti = tqdm(iterable=enumerate(data['info']), total=len(data['info']),
                          desc=f'{self.configuration["dataset_type"]} data for {label} '
                               f'for {self.configuration["model_name"]} is collected')
                for image_idx, each_path in ti:
                    image = self.transform_image(each_path['info'])
                    dataset_dict.append({'idx': each_path['idx'], 'image': image, 'label': label})
                    if self.configuration['uniform']:
                        limit_list = transformation[label]
                        if limit_list[0] != 0:

                            for type_idx, each_limit in enumerate(limit_list):
                                ti.set_description(f'{self.configuration["dataset_type"]} data for {label} for '
                                                   f'{self.configuration["model_name"]} is collected, '
                                                   f'augmentation: {type_idx}')
                                if image_idx < each_limit:
                                    new_image = self.augmentation(image, type_idx)
                                    dataset_dict.append(
                                        {'idx': f'{each_path["idx"]}_a{type_idx}', 'image': new_image, 'label': label})

            with open(self.configuration['output_dir'], 'wb') as out_file:
                pickle.dump(dataset_dict, out_file)
        with open(self.configuration['output_dir'], 'rb') as out_file:
            dataset_dict = pickle.load(out_file)
        if self.configuration['zero_center']:
            zero_center_data = self.zero_center(dataset_dict)
            with open(self.configuration['zero_out_dir'], 'wb') as zero_out:
                pickle.dump(zero_center_data, zero_out)

        out_file = self.configuration['zero_out_dir'] if zero else self.configuration['output_dir']
        with open(out_file, 'rb') as out_data:
            dataset_dict = pickle.load(out_data)

        return dataset_dict

    @staticmethod
    def augmentation(image: np.array, type_idx: int) -> np.array:
        """
        Method is used for augmentation process. Here we have 3 types of augmentation methods: Rotation, Up-Down and
        Left-Right flipping
        :param image: image as np array
        :param type_idx: type of augmentation
        :return: transformed image
        """
        if type_idx == 0:
            new_image = rotate(image, 45)
        elif type_idx == 1:
            new_image = np.flipud(image)
        elif type_idx == 2:
            new_image = np.fliplr(image)
        else:
            raise IndexError('We did not implement such augmentation technique! Do it manually please!')
        return new_image

    def transform_image(self, image_path: str) -> np.array:
        """
        Method loads image and transformed it into desired shape according to the provided image dimension
        :param image_path: path to the requested data
        :return: image in form of array with shape of (1, width, height)
        """
        image = io.imread(image_path)
        if len(image.shape) > 2:
            image = rgb2gray(image)
        resize_img = resize(image, (self.configuration['desired_dim'], self.configuration['desired_dim']))
        reshaped_image = np.reshape(resize_img,
                                    (1, self.configuration['desired_dim'], self.configuration['desired_dim']))

        return reshaped_image

    def zero_center(self, dataset: list) -> list:
        """
        Method is used to apply zero centering to the provided dataset
        :param dataset: original dataset which is collected in the first place and processed
        :return: same form of dataset; only difference is data is zero-centered
        """
        print('Zero centering was activated!')
        dataset_array = np.ndarray(shape=(len(dataset), self.configuration['desired_dim'],
                                          self.configuration['desired_dim']))
        ti = tqdm(enumerate(dataset), total=len(dataset), desc='Zero centering is in progress')
        for idx, data in ti:
            dataset_array[idx] = np.reshape(
                data['image'], (self.configuration['desired_dim'], self.configuration['desired_dim'])
            )

        mean = dataset_array.mean(axis=0)
        dataset_array -= mean
        result_dataset = [
            {'idx': data['idx'], 'image': np.reshape(dataset_array[idx],
                                                     (1, self.configuration['desired_dim'],
                                                      self.configuration['desired_dim'])),
             'label': data['label']} for idx, data in enumerate(dataset)
        ]
        mean_dict = {'mean': mean}
        mean_file = os.path.join(self.configuration['data_env'],
                                 f"dataset_mean_{self.configuration['dataset_type']}.pickle")
        with open(mean_file, 'wb') as mean_data:
            pickle.dump(mean_dict, mean_data)

        return result_dataset

    def __iter__(self):
        """
        Method enables iteration over dataset
        :return: yields each data
        """
        for each in self.dataset:
            yield each
