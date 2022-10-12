import torch.nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utilities import *

class Train:
    def __init__(self, model, model_name, epochs, optimizer, model_params):
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = CrossEntropyLoss()
        self.model_params = model_params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, train_dataset, test_dataset, batch_size, model_file_name, results_file_name):
        train_losses = list()
        train_accuracies = list()
        test_losses = list()
        test_accuracies = list()

        for each_epoch in range(self.epochs):
            epoch_loss = 0
            train_loader, test_loader = generate_loaders(train_dataset, test_dataset, batch_size)
            train_tqdm = tqdm(desc=f'Training epoch {self.model_name}: NaN',
                              total=len(train_loader),
                              iterable=train_loader,
                              leave=True)
            self.model.train()
            total_correct = 0
            for each_batch in train_tqdm:
                train_data = each_batch['data'].to(self.device)
                train_labels = each_batch['labels'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(train_data)
                loss = self.loss(outputs, train_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                train_tqdm.desc = f'Training epoch: {epoch_loss:.4f}'

                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == train_labels).sum().item()

            test_accuracy, test_loss = self.test(test_loader)
            train_tqdm.desc = f'Training epoch {self.model_name}: {each_epoch}'
            train_accuracy = total_correct / len(train_loader.dataset)
            average_train_loss = epoch_loss / len(train_loader)

            print(f'Epoch: {each_epoch}, '
                  f'Train Loss: {average_train_loss :.4f}, '
                  f'Train Accuracy: {train_accuracy :.4f}, '
                  f'Test Loss: {test_loss:.4f}, '
                  f'Test Accuracy: {test_accuracy:.4f}')

            train_losses.append(average_train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            save_model(model_file_name, self.model, each_epoch)
            save_results(train_losses, train_accuracies, test_losses, test_accuracies, results_file_name)
        return train_losses, train_accuracies, test_losses, test_accuracies

    def test(self, test_dataset):
        correct = 0
        total = 0
        losses = 0
        with torch.no_grad():
            self.model.eval()
            for each_set in test_dataset:
                test_data = each_set['data'].to(self.device)
                test_labels = each_set['labels'].to(self.device)
                outputs = self.model(test_data)
                loss = self.loss(outputs, test_labels)
                _, predicted = torch.max(outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
                losses += loss.item()
        average_loss = losses / len(test_dataset)
        accuracy_percent = correct / len(test_dataset.dataset)
        return accuracy_percent, average_loss