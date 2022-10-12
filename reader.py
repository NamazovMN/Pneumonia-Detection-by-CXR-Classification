import numpy as np
import os
from tqdm import tqdm

class ReadData:
    def __init__(self, width, data_path, save_path):
        self.width = width
        self.data_path = data_path
        self.save_path = save_path

    def collect_data(self, model_type):
        folder_path = os.path.join(self.data_path, model_type)
        labels = os.listdir(folder_path)
        data_names_dict = dict()
        for each_label in labels:
            image_folder_path = os.path.join(folder_path, each_label)
            data_names_dict[each_label] = [os.path.join(image_folder_path, each) for each in os.listdir(image_folder_path)]

        return data_names_dict

    def extract_vb(self, data_names_dict):
        images = data_names_dict['PNEUMONIA']
        viruses = list()
        bacterias = list()
        for each_image in images:
            if 'virus' in each_image:
                viruses.append(each_image)
            else:
                bacterias.append(each_image)
        data_names_dict['VIRUS'] = viruses
        data_names_dict['BACTERIA'] = bacterias

        return data_names_dict




