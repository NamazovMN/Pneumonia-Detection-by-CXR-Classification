import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CXRDataset
from src.cnns import BuildModelStructure, CNNModel
from torch import nn
from torch.optim import Adam, SGD
import torch
from sklearn.metrics import f1_score

import os


class RunModelManager:
    """
    Class is used as main performance object which calls required classes and request them to perform their tasks.
    Additionally, after collecting all required data it trains and evaluates the model
    """

    def __init__(self, config_parameters: dict, model_type: str):
        """
        Method is used as initializer of the class
        :param config_parameters: Configuration parameters of the project
        :param model_type: name of the model
        """
        self.model_name = model_type
        self.configuration = self.set_configuration(config_parameters)
        self.model = self.set_model()
        self.loss_fn = self.set_loss_fn()
        self.optimizer = self.set_optimizer(config_parameters)

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

    @staticmethod
    def set_loss_fn() -> nn.CrossEntropyLoss:
        """
        Method is used to set loss function according to the provided parameters
        :return: Loss function
        """
        return nn.CrossEntropyLoss()

    def set_configuration(self, parameters: dict) -> dict:
        """
        Method is used to collect task specific parameters from the provided project relevant parameters
        :param parameters: Configuration parameters of the project
        :return: dictionary which includes task-specific parameters
        """
        configuration = parameters
        configuration["environment"] = os.path.join("train_results",
                                                    f"experiment_{parameters['exp_num']}/{self.model_name}")
        self.check_dir(configuration["environment"])
        checkpoints_dir = os.path.join(configuration["environment"], f"checkpoints")
        self.check_dir(checkpoints_dir)
        configuration["checkpoints"] = checkpoints_dir
        inference_dir = os.path.join(configuration["environment"], f"inferences")
        self.check_dir(inference_dir)
        configuration['inference_dir'] = inference_dir
        return configuration

    def get_loader(self, zero: bool) -> dict:
        """
        Method is used for collecting data loaders for each model in one dictionary
        :param zero: boolean variable specifies whether zero centering will be applied or not
        :return: dictionary which consists of train and test loaders
        """
        train = DataLoader(
            CXRDataset(self.configuration, 'train', self.model_name, zero),
            batch_size=self.configuration['batch_size'],
            shuffle=True
        )

        test = DataLoader(
            CXRDataset(self.configuration, 'test', self.model_name, zero),
            batch_size=self.configuration['batch_size'],
            shuffle=False
        )

        return {'train': train, 'test': test}

    def set_model(self) -> CNNModel:
        """
        Method is used to define specific model according to the requirements
        :return: CNN model which will be used for classification task
        """
        model_structure = BuildModelStructure(self.configuration, self.model_name)
        cnn = CNNModel(model_structure).to(self.configuration['device'])
        return cnn

    def set_optimizer(self, parameters: dict):
        """
        Method is used to set optimizer according to the provided parameters
        :param parameters: configuration parameters include all relevant data for the process
        :return: Optimizer for the model
        """
        if parameters['optimizer'] == 'Adam':
            optimizer = Adam(params=self.model.parameters(), lr=parameters[f'{self.model_name}_learning_rate'])
        elif parameters['optimizer'] == 'SGD':
            optimizer = SGD(params=self.model.parameters(), lr=parameters[f'{self.model_name}_learning_rate'],
                            momentum=0.8)
        else:
            raise Exception('There is not such optimizer in our scenarios. You should choose one of SGD or Adam')
        return optimizer

    @staticmethod
    def compute_accuracy(output: torch.FloatTensor, targets: torch.LongTensor) -> tuple:
        """
        Method is used to compute accuracy based on given target and prediction values
        :param output: tensor data for the output of the model
        :param targets: tensor data for the target values
        :return: tuple that specifies number of correctly predicted data per batch, number of data, predictions
        """
        predicted = torch.argmax(output, dim=-1).tolist()
        targets = targets.view(-1).tolist()
        correct = sum([p == t for p, t in zip(predicted, targets)])
        return correct, len(targets), predicted

    def train_step(self, batch_data: dict) -> tuple:
        """
        Method is used for training the model per batch
        :param batch_data: dictionary contains data and label information for batch
        :return: tuple that contains loss value, accuracy and number of data per batch
        """
        data = batch_data['data']['data']
        self.optimizer.zero_grad()
        output = self.model(data.to(self.configuration['device']))
        output = output.view(-1, output.shape[-1])
        labels = batch_data['label'].view(-1)

        loss = self.loss_fn(output, labels.to(self.configuration['device']))

        loss.backward()
        self.optimizer.step()
        accuracy, num_data, _ = self.compute_accuracy(output, batch_data['label'])
        return loss.item(), accuracy, num_data

    def train_model(self, zero: bool) -> None:
        """
        Method is used for performing whole training procedure
        :param zero: boolean variable specifies whether zero centering will be activated or not
        :return: None
        """
        dataloaders = self.get_loader(zero)

        num_batches = len(dataloaders['train'])
        chosen_epoch = -1
        if self.configuration[f'resume_training_{self.model_name}']:
            chosen_epoch = self.get_epoch(is_best=False)
        init_epoch = 0 if chosen_epoch == -1 else chosen_epoch
        for epoch in range(init_epoch, self.configuration[f'epochs_{self.model_name}']):
            self.model.train()
            epoch_loss = 0
            epoch_accuracy = 0
            num_data = 0
            ti = tqdm(dataloaders['train'], total=num_batches, desc=f'Training epoch {epoch}')
            for batch in ti:
                step_loss, step_accuracy, step_size = self.train_step(batch)
                epoch_loss += step_loss
                epoch_accuracy += step_accuracy
                num_data += step_size
                ti.set_description(f'Epoch {epoch} Training => '
                                   f'Train Loss: {epoch_loss / num_batches: .4f} '
                                   f'Train Accuracy: {epoch_accuracy / num_data: .4f}')
            dev_loss, dev_acc, dev_f1 = self.evaluate(dataloaders['test'], epoch)
            epoch_dict = {
                'train_loss': epoch_loss / num_batches,
                'dev_loss': dev_loss,
                'train_accuracy': epoch_accuracy / num_data,
                'dev_accuracy': dev_acc,
                'f1_dev': dev_f1
            }
            print(f'Epoch: {epoch} F1-score: {dev_f1}')
            self.save_model_details(epoch, epoch_dict)

    def evaluate(self, dev_loader: DataLoader, epoch: int) -> tuple:
        """
        Method is used for evaluating the model at each epoch
        :param dev_loader: data loader object for test data (it is actually test data, since dataset does not include
                           validation dataset
        :param epoch: specifies which epoch is running
        :return: tuple of accuracy, loss and f1 score on validation dataset
        """
        avg_loss = 0
        avg_accuracy = 0
        num_dev_data = 0
        self.model.eval()
        num_batches = len(dev_loader)
        ti = tqdm(dev_loader, total=num_batches, desc=f'Epoch: {epoch} Evaluation =>')
        inference_dict = {'idx': list(), 'prediction': list(), 'target': list()}
        for batch in ti:
            indexes = batch['data']['idx']
            output = self.model(batch['data']['data'].to(self.configuration['device']))
            output = output.view(-1, output.shape[-1])
            labels = batch['label'].view(-1)
            loss = self.loss_fn(output, labels.to(self.configuration['device']))
            step_dev_acc, step_dev_size, predictions = self.compute_accuracy(output, labels)
            avg_accuracy += step_dev_acc
            avg_loss += loss.item()
            num_dev_data += step_dev_size
            ti.set_description(f'Epoch: {epoch} Evaluation => Validation Loss: {avg_loss / num_batches: .4f}, '
                               f'accuracy : {avg_accuracy / num_dev_data :.4f}')
            inference_dict['idx'].extend(indexes)
            inference_dict['prediction'].extend(predictions)
            inference_dict['target'].extend(labels.tolist())
        dev_f1 = f1_score(inference_dict['target'], inference_dict['prediction'], average='macro')
        inference_file = os.path.join(self.configuration['inference_dir'], f'inferences_{epoch}.pickle')
        with open(inference_file, 'wb') as output:
            pickle.dump(inference_dict, output)
        return avg_loss / num_batches, avg_accuracy / num_dev_data, dev_f1

    def save_model_details(self, epoch: int, epoch_dict: dict) -> None:
        """
        Method is used to save model specific data such as training/evaluation results, model and optimizer states per
        epoch
        :param epoch: specifies which epoch is running
        :param epoch_dict: dictionary includes training and validation performance results
        :return:
        """
        results_dict_file = os.path.join(self.configuration['environment'],
                                         f'{self.model_name}_{self.configuration["exp_num"]}_results.pickle')
        if not os.path.exists(results_dict_file):
            result = {
                epoch: epoch_dict
            }
        else:
            with open(results_dict_file, 'rb') as results_dict:
                result = pickle.load(results_dict)
            result[epoch] = epoch_dict

        with open(results_dict_file, 'wb') as result_dict:
            pickle.dump(result, result_dict)

        model_dict_name = os.path.join(self.configuration['checkpoints'],
                                       f"epoch_{epoch}_{self.model_name}_dev_loss{epoch_dict['dev_loss']: .3f}"
                                       f"_train_loss_{epoch_dict['train_loss']: .3f}"
                                       f"_dev_accuracy_{epoch_dict['dev_accuracy']: .3f}"
                                       f"_f1_{epoch_dict['f1_dev']: .3f}")
        optimizer_dict_name = os.path.join(self.configuration['checkpoints'], f'optim_epoch_{epoch}')
        torch.save(self.model.state_dict(), model_dict_name)
        torch.save(self.optimizer.state_dict(), optimizer_dict_name)
        print(f'Model and optimizer parameters for epoch {epoch} were saved successfully!')
        print(f'{20 * "<"}{20 * ">"}')

    def get_epoch(self, is_best: bool) -> int:
        """
        Method is used to get epoch according to the specified boolean variable. If it is true it returns the best epoch
        according to the f1 scores in epochs, otherwise it returns the last epoch was trained
        :param is_best: boolean variable specifies the best one is requested or not
        :return: requested epoch / -1 in case no training has been done
        """
        result_file = os.path.join(self.configuration['environment'],
                                   f"{self.model_name}_{self.configuration['exp_num']}_results.pickle")
        if not os.path.exists(result_file):
            print('No results file was found! It happens because there is not any trained data!')
            chosen = -1
        else:
            with open(result_file, 'rb') as result_data:
                result_dict = pickle.load(result_data)
            if is_best:
                f1_results = dict()

                for epoch, epoch_dict in result_dict.items():
                    f1_results[epoch] = epoch_dict['f1_dev']
                print('HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEY')
                print(f1_results)
                chosen = max(f1_results, key=f1_results.get)
                print(f1_results[chosen])
                print(chosen)
            else:
                chosen = max(result_dict.keys())

        return chosen

    def load_model(self, is_best: bool = False) -> None:
        """
        Method is used to load the model according to is_best value
        :param is_best: boolean variable specifies the best one is requested or not
        :return: None
        """
        epoch = self.get_epoch(is_best)
        file_names = {'model_path': str(), 'optim_path': str()}
        for file_name in os.listdir(self.configuration['checkpoints']):
            if f'epoch_{epoch}_' in file_name:
                if self.model_name in file_name:
                    file_names['model_path'] = os.path.join(self.configuration['checkpoints'], file_name)
                    file_names['optim_path'] = os.path.join(self.configuration['checkpoints'], f'optim_epoch_{epoch}')
                # elif 'optim' in file_name:
                #     file_names['optim_path'] = os.path.join(self.configuration['checkpoints'], file_name)

        if file_names['model_path']:
            print(f"Model is loaded from {file_names['model_path']}")
            self.model.load_state_dict(torch.load(file_names['model_path'], map_location=self.configuration['device']))
            self.optimizer.load_state_dict(
                torch.load(file_names['optim_path'], map_location=self.configuration['device']))
            self.model.eval()
        else:
            raise FileNotFoundError('No trained model was found')
