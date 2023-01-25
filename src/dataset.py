import numpy as np
import torch
from torch.utils.data import Dataset
from src.collect_data import CollectDataset


class CXRDataset(Dataset):
    """
    Class is a dataset object
    """

    def __init__(self, config_parameters: dict, dataset_type: str, model_type: str, zero: bool):
        """
        Method is used as initializer of the class
        :param config_parameters: Configuration parameters of the project
        :param dataset_type: type of dataset which can either be training or test dataset
        :param model_type: name of the model
        :param zero: specifies whether zero centering will be done or not
        """
        self.ds_collector = CollectDataset(config_parameters, dataset_type, model_type, zero)
        self.idx, self.data, self.labels = self.collect_dataset()

    def collect_dataset(self) -> tuple:
        """
        Method is used to collect dataset and return it in desired format
        :return: tuple which includes:
                list of indexes;
                float tensor for data
                long tensor for labels
        """
        data = list()
        labels = list()
        indexes = list()
        label2id = self.ds_collector.labels
        for each in self.ds_collector:
            indexes.append(each['idx'])
            data.append(each['image'])
            labels.append(label2id[each['label']])
        return indexes, torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels))

    def __getitem__(self, idx: int) -> dict:
        """
        Method is used as getter to get specific data according to its index in dataset. Notice that our indexes are
        different. They will be used as flag in further steps
        :param idx: index specify place of the data in the dataset
        :return: dictionary which:
                data: dictionary for index and image
                label: image's label
        """
        return {
            'data': {'idx': self.idx[idx], 'data': self.data[idx]},
            'label': self.labels[idx]
        }

    def __len__(self) -> int:
        """
        Method is used for computing length of the Dataset
        :return: integer specifies number of data in this specific dataset
        """
        return len(self.data)
