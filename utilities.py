import json
import os
from collections import Counter
import torch
from torch.utils.data import DataLoader
from DatasetGen import Xrays
import matplotlib.pyplot as plt


def count_data(dataset, model_name, labels):

    print(f'For {model_name} number of data according to the labels are:')
    given_dataset = [dataset[model_name]['train'], dataset[model_name]['test']]
    datasets = ['TRAIN', 'TEST']
    for index, each_dataset in enumerate(given_dataset):
        print(f'{datasets[index]}: ')
        counter_labels = Counter(each_dataset['labels'])
        for idx, each_label in enumerate(labels):
            for each_info in counter_labels.most_common():
                if idx == each_info[0]:
                    print(f'Label: {each_label}; Number of data: {each_info[1]}; Idx: {idx}')

        print('=============================================================================')
    print('=============================================================================')


def generate_dataset(dataset, model_name):

    train_dataset = Xrays(dataset[model_name]['train'])
    test_dataset = Xrays(dataset[model_name]['test'])
    # val_dataset = Xrays(dataset[model_name]['val'])

    return train_dataset, test_dataset


def generate_loaders(train_dataset, test_dataset, batch_size, shuffle=True):
    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size)
    # val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size)

    return train_loader, test_loader


def compute_output(model_params, num_layers):
    kernels = [model_params[f'kernel{k + 1}'] for k in range(num_layers)]
    mp_kernels = [model_params[f'mp_kernel{k + 1}'] for k in range(num_layers)]
    strides = [model_params[f'stride{k + 1}'] for k in range(num_layers)]
    mp_strides = [model_params[f'mp_stride{k + 1}'] for k in range(num_layers)]
    out = model_params['input_size']
    for each in range(len(kernels)):
        out = ((out - kernels[each]) / strides[each]) + 1
        out = ((out - mp_kernels[each]) / mp_strides[each]) + 1

    return int(out)


def model_params_om(width, experience, uniform=True):
    mp_om = {'input_size': width,
             'experience': experience,
             'uniform': uniform,
             'in_size': 1,

             'kernel1': 5,
             'kernel2': 4,
             'kernel3': 4,
             # 'kernel4': 2,

             'stride1': 2,
             'stride2': 2,
             'stride3': 2,
             # 'stride4': 1,

             'mp_kernel1': 2,
             'mp_kernel2': 2,
             'mp_kernel3': 2,
             # 'mp_kernel4': 3,

             'mp_stride1': 2,
             'mp_stride2': 2,
             'mp_stride3': 1,
             # 'mp_stride4': 1,

             'out1': 30,
             'out2': 60,
             'out3': 120,
             # 'out4': 120,

             'dp1': 0.2,
             'dp2': 0.3,
             'dp3': 0.5,

             'linout1': 100,
             'linout2': 30,
             # 'linout3': 10,
             'out_size': 3,

             'isbn1': True,
             'isbn2': True,
             'isbn3': False
             # 'isbn4': False

             }
    return mp_om

def model_params_np(width, experience, uniform=True):
    mp_np = {'input_size': width,
             'experience': experience,
             'uniform': uniform,

             'in_size': 1,

             'kernel1': 5,
             'kernel2': 3,
             'kernel3': 3,

             'stride1': 2,
             'stride2': 1,
             'stride3': 1,

             'mp_kernel1': 3,
             'mp_kernel2': 3,
             'mp_kernel3': 2,

             'mp_stride1': 2,
             'mp_stride2': 2,
             'mp_stride3': 1,

             'out1': 30,
             'out2': 60,
             'out3': 112,

             'dp1': 0.2,
             'dp2': 0.4,
             'dp3': 0.5,

             'linout1': 100,
             'linout2': 50,
             'linout3': 24,
             'out_size': 2,

             'isbn1': True,
             'isbn2': True,
             'isbn3': False,

             }
    return mp_np


def model_params_vb(width, experience, uniform=True):
    mp_vb = {'input_size': width,
             'experience': experience,
             'uniform': uniform,
             'in_size': 1,
             'kernel1': 5,
             'kernel2': 4,
             'kernel3': 4,
             # 'kernel4': 2,
             'stride1': 2,
             'stride2': 2,
             'stride3': 2,
             # 'stride4': 1,
             'mp_kernel1': 2,
             'mp_kernel2': 2,
             'mp_kernel3': 2,
             # 'mp_kernel4': 2,
             'mp_stride1': 2,
             'mp_stride2': 2,
             'mp_stride3': 1,
             # 'mp_stride4': 1,
             'out1': 26,
             'out2': 52,
             'out3': 86,
             # 'out4': 160,
             'dp1': 0.4,
             'dp2': 0.3,
             'dp3': 0.7,
             'linout1': 100,
             'linout2': 50,
             'out_size': 2,
             'isbn1': True,
             'isbn2': True,
             'isbn3': False
             # 'isbn4': False
             }
    return mp_vb


def save_params(model_params, experiment_num, experiment_path):
    file_name = os.path.join(experiment_path, f'model_params_{experiment_num}.json')
    with open(file_name, 'w') as params:
        json.dump(model_params, params)

# def save_params(model_name, model_params, save_path):
#     experience = model_params['experience']
#     folder_model = os.path.join(save_path, f'{model_name}_{experience}')
#
#     if not os.path.exists(folder_model):
#         os.makedirs(folder_model)
#     file_name = os.path.join(folder_model, f'{model_name}_{experience}_params.json')
#     with open(file_name, 'w') as params:
#         json.dump(model_params, params)


# def save_model(model_name, save_path, model, epoch_num, model_params):
#     experience = model_params['experience']
#     folder_model = os.path.join(save_path, f'{model_name}_{experience}')
#     if not os.path.exists(folder_model):
#         os.makedirs(folder_model)
#     path_for_model = os.path.join(folder_model, f'{model_name}_{experience}_{epoch_num}')
#     torch.save(model.state_dict(), path_for_model)

def save_model(file_name, model, epoch_num):
    model_path = file_name + f'_{epoch_num}'
    torch.save(model.state_dict(), model_path)


def save_results(tr_loss, tr_acc, test_loss, test_acc, file_name):
    results = {
        'tr_loss': tr_loss,
        'tr_acc': tr_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    with open(file_name, 'w') as new_file:
        json.dump(results, new_file)


def load_results(file_name):
    with open(file_name) as json_file:
        results = json.load(json_file)
    return results


def plot_and_save(results_dict, model_name, num_epoch, graph_loss, graph_acc):
    loss_tr = results_dict['tr_loss'][:num_epoch]
    loss_test = results_dict['test_loss'][:num_epoch]
    acc_tr = results_dict['tr_acc'][:num_epoch]
    acc_test = results_dict['test_acc'][:num_epoch]
    epochs = list(range(num_epoch))
    # print(epochs)
    plot_loss = plt.figure(1)
    plt.plot(epochs, loss_tr)
    plt.plot(epochs, loss_test)
    plt.title(f'Train and Test loss graphs of {model_name} model')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend(['Train', 'Test'])
    plt.savefig(graph_loss)
    plt.show()
    plot_acc = plt.figure(2)
    plt.plot(epochs, acc_tr)
    plt.plot(epochs, acc_test)
    plt.title(f'Train and Test accuracy graphs of {model_name} model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.savefig(graph_acc)
    plt.show()