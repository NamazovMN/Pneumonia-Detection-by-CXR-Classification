from utilities import *
from reader import ReadData
from generate_Dataset import DataPrep
from models import CNN_model_om, CNN_model_np, CNN_model_vb
from training import Train
from torch.optim import Adam, SGD
from precision import Precision

main_path = 'data'
experiment_folder = 'experiments'

uniform = True
dist_type = 'uniform' if uniform else 'non-uniform'
print(f'Data distribution set to {dist_type}!')
print('============================')
save_path = 'saved_data_uniform' if uniform else 'saved_data_nonuniform'
width = 150
height = 150
experiment_num = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reader = ReadData(width, 'data', save_path)
train_dataset = reader.collect_data('train')
models_dict = {
    'om': ['NORMAL', 'VIRUS', 'BACTERIA'],
    'np': ['NORMAL', 'PNEUMONIA'],
    'vb': ['VIRUS', 'BACTERIA']
}

batch_size = 64
epochs = 1000
om_lr = 0.0001
vb_lr = 0.0001
np_lr = 0.0002

experiment_path = os.path.join(experiment_folder, f'experiment_{experiment_num}')
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
om_parameters = model_params_om(width, experiment_num, uniform)
vb_parameters = model_params_vb(width, experiment_num, uniform)
np_parameters = model_params_np(width, experiment_num, uniform)

dp = DataPrep(reader, 'data', save_path, width, height)
dict_data = dp.save_and_load(uniform)

om_train_dataset, om_test_dataset = generate_dataset(dict_data, 'om')
np_train_dataset, np_test_dataset = generate_dataset(dict_data, 'np')
vb_train_dataset, vb_test_dataset = generate_dataset(dict_data, 'vb')

count_data(dict_data, 'om', models_dict['om'])
count_data(dict_data, 'np', models_dict['np'])
count_data(dict_data, 'vb', models_dict['vb'])

om_train_loader, om_test_loader = generate_loaders(om_train_dataset, om_test_dataset, batch_size)
np_train_loader, np_test_loader = generate_loaders(np_train_dataset, np_test_dataset, batch_size)
vb_train_loader, vb_test_loader = generate_loaders(vb_train_dataset, vb_test_dataset, batch_size)

all_params = {'om': om_parameters, 'vb': vb_parameters, 'np': np_parameters}
save_params(all_params, experiment_num, experiment_path)

om_folder = os.path.join(experiment_path, f'om_{experiment_num}')
om_loss_graph = os.path.join(experiment_path, f'om_{experiment_num}_loss.png')
om_acc_graph = os.path.join(experiment_path, f'om_{experiment_num}_acc.png')
om_results_file = os.path.join(experiment_path, f'om_{experiment_num}_results.json')

vb_folder = os.path.join(experiment_path, f'vb_{experiment_num}')
vb_loss_graph = os.path.join(experiment_path, f'vb_{experiment_num}_loss.png')
vb_acc_graph = os.path.join(experiment_path, f'vb_{experiment_num}_acc.png')
vb_results_file = os.path.join(experiment_path, f'vb_{experiment_num}_results.json')

np_folder = os.path.join(experiment_path, f'np_{experiment_num}')
np_loss_graph = os.path.join(experiment_path, f'np_{experiment_num}_loss.png')
np_acc_graph = os.path.join(experiment_path, f'np_{experiment_num}_acc.png')
np_results_file = os.path.join(experiment_path, f'np_{experiment_num}_results.json')

if not os.path.exists(om_folder):
    os.makedirs(om_folder)
if not os.path.exists(np_folder):
    os.makedirs(np_folder)
if not os.path.exists(vb_folder):
    os.makedirs(vb_folder)

om_models = os.path.join(om_folder, f'om_{experiment_num}')
vb_models = os.path.join(vb_folder, f'vb_{experiment_num}')
np_models = os.path.join(np_folder, f'np_{experiment_num}')

om_model = CNN_model_om(om_parameters).to(device)
vb_model = CNN_model_vb(vb_parameters).to(device)
np_model = CNN_model_np(np_parameters).to(device)
# only_plot = False
only_plot = True
if only_plot:
    om_results = load_results(om_results_file)
    np_results = load_results(np_results_file)
    vb_results = load_results(vb_results_file)
    # plot_and_save(om_results, 'Direct', 125, om_loss_graph, om_acc_graph)
    # plot_and_save(np_results, 'NP-CNN', 68, np_loss_graph, np_acc_graph)
    # plot_and_save(vb_results, 'VB-CNN', 131, vb_loss_graph, vb_acc_graph)
    # #

    # for i in range(45, 90):
    #     print(f'epoch: {i}')
    om_model.load_state_dict(torch.load(om_models + f'_{88}'))

    np_model.load_state_dict(torch.load(np_models + f'_{64}'))
    vb_model.load_state_dict(torch.load(vb_models + f'_{84}')) #98
    # np_model.load_state_dict(torch.load(np_models + '_68'))
    # vb_model.load_state_dict(torch.load(vb_models + '_99'))
    # #
    precision_compute = Precision(save_path, om_model, vb_model, np_model, om_test_loader, vb_test_loader, np_test_loader,
                              experiment_path, experiment_num, device)
    precision_compute.compute_precision_cascade()
        # print('=============================')

    print('======================================')

else:
    print(60 * '*')
    print('Train session of OM model started!')
    print(60 * '*')

    om_optim = SGD(om_model.parameters(), lr=om_lr, momentum=0.9)
    # om_optim = Adam(om_model.parameters(), lr=om_lr)

    om_train = Train(om_model, 'om', epochs, om_optim, om_parameters)
    # om_tr_loss, om_tr_acc, om_test_loss, om_test_acc = om_train.train(om_train_loader, om_test_loader, om_models,
    #                                                                   om_results_file)
    om_tr_loss, om_tr_acc, om_test_loss, om_test_acc = om_train.train(om_train_dataset, om_test_dataset, batch_size, om_models,
                                                                      om_results_file)

    # print(60 * '*')
    # print('Train session of VB model started!')
    # print(60 * '*')
    #
    # vb_optim = SGD(vb_model.parameters(), lr=vb_lr, momentum=0.8)
    # vb_train = Train(vb_model, 'vb', epochs, vb_optim, vb_parameters)
    # # vb_tr_loss, vb_tr_acc, vb_test_loss, vb_test_acc = vb_train.train(vb_train_loader, vb_test_loader, vb_models,
    # #                                                                   vb_results_file)
    #
    # vb_tr_loss, vb_tr_acc, vb_test_loss, vb_test_acc = vb_train.train(vb_train_dataset, vb_test_dataset, batch_size, vb_models,
    #                                                                   vb_results_file)
    #
    # print(60 * '*')
    # print('Train session of NP model started!')
    # print(60 * '*')
    #
    # # np_optim = Adam(np_model.parameters(), lr=np_lr)
    # np_optim = SGD(np_model.parameters(), lr=np_lr, momentum=0.9)
    # np_train = Train(np_model, 'np', epochs, np_optim, np_parameters)
    # np_tr_loss, np_tr_acc, np_test_loss, np_test_acc = np_train.train(np_train_dataset, np_test_dataset, batch_size, np_models,
    #                                                                   np_results_file)
