import os
import random

from tqdm import tqdm
import cv2
from reader import ReadData
from skimage.transform import resize, rotate
import numpy as np
import textwrap


class DataPrep:
    def __init__(self, reader, main_path, save_path, width, height):
        self.main_path = main_path
        self.save_path = save_path
        self.width = width
        self.height = height
        self.reader = reader

    def transform_img(self, image):
        resize_img = resize(image, (self.width, self.height))
        reshaped_image = np.reshape(resize_img, (1, self.width, self.height))

        return reshaped_image

    def read_Data(self, type_data):
        images_dict = dict()

        names_dict = self.reader.collect_data(type_data)
        names_dict = self.reader.extract_vb(names_dict)
        for each_dataset in names_dict.keys():
            dataset = names_dict[each_dataset]
            images = list()
            dataset_name = f'{each_dataset[0]}{each_dataset[1:].lower()}'
            tqdm_iterator = tqdm(desc=f'{dataset_name} data is collected for {type_data} with shape of {self.width}x{self.height}',
                                 iterable=dataset,
                                 total=len(dataset))
            for each_imn in tqdm_iterator:
                image = cv2.imread(each_imn, 0)
                images.append(self.transform_img(image))

            images_dict[each_dataset] = images
        return images_dict

    def organize_dataset_uniformly(self, dataset, labels, info_sentence):
        num_items = dict()
        labels_info = 'Number of data at each label is as following: \n'
        for label in labels:
            num_items[label] = len(dataset[label])
            labels_info = labels_info + f'{label}: {num_items[label]} '
        max_val = max(list(num_items.values()))
        max_label = [l for l, v in num_items.items() if v == max_val]

        demanded_items = {each_label: max_val - num_items[each_label] for each_label in num_items.keys()}
        print(info_sentence + ' data and labels are augmented and collected!')
        print(labels_info)
        print(30 * '--')
        print(f'Since maximum number of data belongs to the {max_label[0]} '
              f'label with {max_val} data, others will be complemented to that number!')
        print(30 * '**')
        generated_dataset = dict()
        result_sent = 'As a result of data augmentation: '
        for each_label in demanded_items.keys():
            coefficient = int(demanded_items[each_label] / num_items[each_label])
            # print(coefficient)
            if not demanded_items[each_label]:
                generated_dataset[each_label] = dataset[each_label]
            else:
                if not coefficient:
                    rest = random.sample(list(range(num_items[each_label])), demanded_items[each_label])
                else:
                    fully_transform = demanded_items[each_label] - num_items[each_label] * coefficient
                    rest = random.sample(list(range(num_items[each_label])), fully_transform)

                generated_dataset[each_label] = dataset[each_label] + self.generate_dataset(dataset[each_label], coefficient, rest)
            result_sent = result_sent + f'{each_label}: {len(generated_dataset[each_label])} '
        print(result_sent)

        return generated_dataset

    def generate_dataset(self, data, coefficient, rest):
        f_transform_choices = {0: 'flipud', 1: 'fliprl', 2: 'rotate45'}

        rest_data = list()
        for each_idx in rest:
            rest_data.append(rotate(data[each_idx], 90))
        transformed_data = list()

        if coefficient:
            chosen_transforms = random.sample(list(range(0, coefficient)), coefficient)

            print(f'transformation will be applied: {[f_transform_choices[each] for each in chosen_transforms]}')
            for each in chosen_transforms:
                transformation = f_transform_choices[each]
                if transformation == 'flipud':
                    for each_data in data:
                        transformed_data.append(np.flipud(each_data))
                elif transformation == 'fliplr':
                    for each_data in data:
                        transformed_data.append(np.fliplr(each_data))
                elif transformation == 'rotate45':
                    for each_data in data:
                        transformed_data.append(rotate(each_data, 45))

        transformed_data += rest_data

        return transformed_data

    def get_each_dataset(self, dataset, info_sent, labels, uniform=True):
        classes = list()
        data = list()
        if uniform:
            print(f"{20 * '<'}Uniform data preparation started!{20 * '>'}")
            print(f"{30 * '<'}{30 * '>'}")
        dataset_new = self.organize_dataset_uniformly(dataset, labels, info_sent) if uniform else dataset
        print(30*'**')
        print(f'Dataset : {info_sent} includes:')
        for each_key in dataset_new:
            images = dataset_new[each_key]
            print(f'Label {each_key}:   {len(dataset_new[each_key])} images with shape of {images[0].shape}')
        print(30 * '**')
        print(30 * '**')

        for idx, each in enumerate(labels):

            for each_image in dataset_new[each]:
                data.append(each_image)
                classes.append(idx)
        return data, classes

    def collect_dataset(self, uniform):
        train_images = self.read_Data('train')
        test_images = self.read_Data('test')

        om_train_data, om_train_labels = self.get_each_dataset(train_images,
                                                               'One Model (OM) train',
                                                               ['NORMAL', 'VIRUS', 'BACTERIA'], uniform)
        om_test_data, om_test_labels = self.get_each_dataset(test_images,
                                                             'One Model (OM) test',
                                                             ['NORMAL', 'VIRUS', 'BACTERIA'], uniform)

        np_train_data, np_train_labels = self.get_each_dataset(train_images,
                                                               'Normal/Pneumonia (NP) train',
                                                               ['NORMAL', 'PNEUMONIA'], uniform)

        np_test_data, np_test_labels = self.get_each_dataset(test_images,
                                                             'Normal/Pneumonia (NP) test',
                                                             ['NORMAL', 'PNEUMONIA'], uniform)

        vb_train_data, vb_train_labels = self.get_each_dataset(train_images,
                                                               'Virus/Bacteria (VB) train',
                                                               ['VIRUS', 'BACTERIA'], uniform)

        vb_test_data, vb_test_labels = self.get_each_dataset(test_images,
                                                             'Virus/Bacteria (VB) test',
                                                             ['VIRUS', 'BACTERIA'], uniform)

        datasets = {'om': {'train': {'data': np.array(om_train_data), 'labels': np.array(om_train_labels)},
                           'test': {'data': np.array(om_test_data), 'labels': np.array(om_test_labels)}},
                    'np': {'train': {'data': np.array(np_train_data), 'labels': np.array(np_train_labels)},
                           'test': {'data': np.array(np_test_data), 'labels': np.array(np_test_labels)}},
                    'vb': {'train': {'data': np.array(vb_train_data), 'labels': np.array(vb_train_labels)},
                           'test': {'data': np.array(vb_test_data), 'labels': np.array(vb_test_labels)}}
                    }

        return datasets

    def save_and_load(self, uniform):
        models = ['om', 'np', 'vb']
        data_type = ['train', 'test']
        dl = ['data', 'labels']
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            datasets = self.collect_dataset(uniform)
            for each_model in models:
                for each_data_type in data_type:
                    for each_dl in dl:
                        data_file = f'{each_model}_{each_data_type}_{each_dl}.npy'
                        data = datasets[each_model][each_data_type][each_dl]
                        path_file = os.path.join(self.save_path, data_file)
                        np.save(path_file, data)
                    print(f'Data and Labels for {each_data_type} for {each_model} model were saved successfully!')
                    print('========================================================================================')
        else:
            datasets = dict()
            for each_model in models:
                data_type_dict = dict()
                for each_data_type in data_type:
                    dl_dict = dict()
                    for each_dl in dl:
                        data_file = f'{each_model}_{each_data_type}_{each_dl}.npy'
                        path_file = os.path.join(self.save_path, data_file)
                        data = np.load(path_file)
                        dl_dict[each_dl] = data
                    print(f'Data and Labels for {each_data_type} for {each_model} model were loaded successfully!')
                    print('========================================================================================')

                    data_type_dict[each_data_type] = dl_dict
                datasets[each_model] = data_type_dict

        return datasets
