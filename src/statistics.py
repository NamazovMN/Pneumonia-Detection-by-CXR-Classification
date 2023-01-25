from src.collect_info import CollectInformation
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
import skimage
import pickle
import collections
from sklearn.metrics import confusion_matrix
import seaborn as sns
from src.runner import RunModelManager
from src.collect_data import CollectDataset


class Statistics:
    """
    Class is used to run statistics in the project (before and after training)
    """

    def __init__(self, config_parameters: dict, model_type: str):
        """
        Method is used as initializer of the class
        :param config_parameters: Configuration parameters of the project
        :param model_type: name of the model
        """
        self.model_name = model_type
        self.configuration = self.set_configuration(config_parameters)
        self.train_info = CollectInformation(config_parameters, 'train')
        self.test_info = CollectInformation(config_parameters, 'test')
        self.runner = RunModelManager(config_parameters, model_type)

    def set_configuration(self, parameters: dict) -> dict:
        """
        Method is used to collect task specific parameters from the provided project relevant parameters
        :param parameters: Configuration parameters of the project
        :return: dictionary which includes task-specific parameters
        """
        configuration = parameters
        experiment_environment = os.path.join('train_results', f"experiment_{parameters['exp_num']}/{self.model_name}")
        results_file = os.path.join(experiment_environment, f"{self.model_name}_{parameters['exp_num']}_results.pickle")
        configuration['environment'] = experiment_environment
        configuration['results'] = results_file
        data_collect = CollectDataset(parameters, 'test', self.model_name)
        labels_dict = data_collect.get_labels()
        configuration['labels'] = labels_dict
        configuration['id2label'] = self.reverse_labels(labels_dict)
        return configuration

    @staticmethod
    def reverse_labels(labels_dict: dict) -> dict:
        """
        Method is used for reverting labels data for idx to label processes
        :param labels_dict: dictionary in which keys are labels and values are their indexes
        :return: dictionary object in which keys are indexes and values are labels
        """
        return {idx: label for label, idx in labels_dict.items()}

    def plot_bar(self, info_dict: dict, is_train: bool = True) -> None:
        """
        Method is used to plot bar chart for number of data per label in original dataset
        :param info_dict: information dictionary per model
        :param is_train: boolean variable specifies whether train or test dataset is analyzed
        :return: None
        """
        train_info = 'train' if is_train else 'test'
        plt.figure()
        title = f"Data distribution for labels in {train_info} dataset of " \
                f"{self.model_name} model"
        plt.title(title)
        plt.xticks(np.arange(len(info_dict.keys())))
        plt.bar(info_dict.keys(), info_dict.values())
        plt.plot()
        figure_path = os.path.join(self.train_info.configuration['info_dir'],
                                   f'{self.model_name}_{train_info}_dist.png')
        plt.savefig(figure_path)

    def distribution(self):
        """
        Method is used for collecting required information per model and call plot function to visualize distribution
        along with printing the data distribution in dataset
        :return:
        """

        plot_info = dict()
        for label, val in self.train_info[self.model_name].items():
            print(f"Number of Chest X-Ray images train dataset per {label}: {val['num_data']}")
            plot_info[label] = val['num_data']
        self.plot_bar(plot_info, is_train=True)
        for label, val in self.test_info[self.model_name].items():
            print(f"Number of Chest X-Ray images in test dataset per {label}: {val['num_data']}")
            plot_info[label] = val['num_data']
        self.plot_bar(plot_info, is_train=False)
        print(f"{20 * '<'}{20 * '>'}")

    def provide_examples(self) -> None:
        """
        Method is used to visualize examples for each label
        :return: None
        """
        num_images = 3
        num_labels = len(self.train_info[self.model_name].keys())

        figure, axis = plt.subplots(num_labels, num_images, figsize=(20, 10))
        figure.suptitle(f'Example images for labels in {self.model_name.upper()} dataset')

        for label_idx, (label, val) in enumerate(self.train_info[self.model_name].items()):
            for idx in range(num_images):
                axis[label_idx, idx].set_title(label)
                image = io.imread(val['info'][idx]['info'])
                image = skimage.transform.resize(image, (800, 800))
                axis[label_idx, idx].imshow(image)
                # axis[label_idx][idx] = io.imread(val['info'][idx])
                plt.plot()
                plt.grid()
        figure_path = os.path.join(self.train_info.configuration['info_dir'], f'{self.model_name}_examples.png')
        figure.savefig(figure_path)

    def plot_results(self, is_accuracy: bool = True) -> None:
        """
        Method is used to plot accuracy/loss graphs after training session is over, according to provided variable
        :param is_accuracy: boolean variable specifies the type of data will be plotted
        :return: None
        """

        metric_key = 'accuracy' if is_accuracy else 'loss'
        dev_data = list()
        train_data = list()
        with open(self.configuration['results'], 'rb') as result_data:
            result_dict = pickle.load(result_data)
        ordered = collections.OrderedDict(sorted(result_dict.items()))

        for epoch, results in ordered.items():
            dev_data.append(results[f'dev_{metric_key}'])
            train_data.append(results[f'train_{metric_key}'])
        plt.figure()
        plt.title(f'{metric_key.title()} results over {len(result_dict.keys())} epochs for {self.model_name.upper()}')
        plt.plot(list(result_dict.keys()), train_data, 'g', label='Train')
        plt.plot(list(result_dict.keys()), dev_data, 'r', label='Validation')
        plt.grid()
        plt.xlabel('Number of epochs')
        plt.ylabel(f'{metric_key.title()} results')
        plt.legend(loc=4)
        figure_path = os.path.join(self.configuration['environment'], f'{metric_key}_plot.png')
        plt.savefig(figure_path)
        plt.show()

    def get_confusion_matrix(self) -> None:
        """
        Method is used to generate confusion matrix and call specific method to visualize it
        :return: None
        """
        best_epoch = self.runner.get_epoch(is_best=True)
        file_name = os.path.join(self.configuration['environment'], f'inferences/inferences_{best_epoch}.pickle')
        with open(file_name, 'rb') as inference_dict:
            inference_data = pickle.load(inference_dict)
        targets = [self.configuration['id2label'][idx] for idx in inference_data['target']]
        predictions = [self.configuration['id2label'][idx] for idx in inference_data['prediction']]
        conf_matrix = confusion_matrix(targets, predictions)
        self.plot_confusion_matrix(conf_matrix)

    def plot_confusion_matrix(self, conf_matrix: np.array, cascade: bool = False) -> None:
        """
        Method is used for visualization of provided confusion matrix
        :param conf_matrix: numpy array which expresses confusion matrix
        :param cascade: boolean variable which specifies confusion matrix is generated for cascade operation or not
        :return: None
        """

        cascade_info = '_cascade' if cascade else ''
        labels = [self.configuration['id2label'][idx] for idx in range(len(self.configuration['labels']))]
        plt.figure(figsize=(8, 6), dpi=100)
        sns.set(font_scale=1.1)

        ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )

        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=20)
        ax.xaxis.set_ticklabels(labels)

        ax.set_ylabel("Actual Labels", fontsize=14, labelpad=20)
        ax.yaxis.set_ticklabels(labels)
        model_name = 'Cascade' if cascade else self.model_name.upper()
        ax.set_title(f"Confusion Matrix for {model_name}", fontsize=14, pad=20)
        image_name = os.path.join(self.configuration['environment'], f'confusion_matrix{cascade_info}.png')
        plt.savefig(image_name)
        plt.show()

    def provide_statistics(self, before: bool = True) -> None:
        """
        Method is used to provide all required statistics
        :param before: boolean variable to specify which statistics is required
        :return: None
        """
        if before:
            self.distribution()
            self.provide_examples()
        else:
            self.plot_results()
            self.plot_results(False)
            self.get_confusion_matrix()
