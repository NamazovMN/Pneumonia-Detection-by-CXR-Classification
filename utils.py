import argparse

import torch.cuda


def set_configuration() -> argparse.Namespace:
    """
    Function is used to collect project parameters from the user. User can provide any of these variables
    :return: parser object which contain all parameters are required for the project
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', default='datasets', required=False, type=str,
                        help='Specifies dataset directory')
    parser.add_argument('--batch_size', default=32, required=False, type=int,
                        help='Specifies number of data that each batch will contain')
    parser.add_argument('--om_learning_rate', default=0.0001, required=False, type=float,
                        help='Specifies learning rate for the om model')
    parser.add_argument('--ph_learning_rate', default=0.00001, required=False, type=float,
                        help='Specifies learning rate for the ph model')
    parser.add_argument('--vb_learning_rate', default=0.0005, required=False, type=float,
                        help='Specifies learning rate for the ph model')

    parser.add_argument('--om_out', default=[64, 128, 192], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of OneModel')
    parser.add_argument('--om_kernels', default=[5, 4, 3], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of OneModel')
    parser.add_argument('--om_strides', default=[2, 2, 1], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of OneModel')
    parser.add_argument('--om_pooling_details', default=[(2, 2), (2, 2), (2, 1)], required=False, type=tuple, nargs='+',
                        help='Specifies kernel dimensions of OneModel')
    parser.add_argument('--om_bn_apply', default=[True, False, False], required=False, type=bool, nargs='+',
                        help='Specifies kernel dimensions of OneModel')
    parser.add_argument('--om_drop_cnn', default=[0.2, 0.3, 0.4], required=False, type=float, nargs='+',
                        help='Specifies kernel dimensions of OneModel')
    parser.add_argument('--om_drop_lin', default=[0.3], required=False, type=float, nargs='+',
                        help='Specifies kernel dimensions of OneModel')
    parser.add_argument('--om_lin_out', default=[110], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--om_bn_lin_apply', default=[False], required=False, type=bool, nargs='+',
                        help='Specifies batch normalization application for FCN of OneModel')
    parser.add_argument('--om_train', default=False, required=False, action='store_true')

    parser.add_argument('--vb_out', default=[200, 300, 400], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--vb_kernels', default=[5, 4, 3], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--vb_strides', default=[2, 2, 1], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--vb_pooling_details', default=[(2, 2), (2, 2), (2, 2)], required=False, type=tuple, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--vb_bn_apply', default=[True, False, False], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--vb_drop_cnn', default=[0.3, 0.4, 0.5], required=False, type=float, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--vb_lin_out', default=[200, 50], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--vb_drop_lin', default=[0.3, 0], required=False, type=float, nargs='+',
                        help='Specifies kernel dimensions of Virus-Bacteria Model')
    parser.add_argument('--vb_bn_lin_apply', default=[True, False], required=False, type=int, nargs='+',
                        help='Specifies batch normalization application for FCN of Virus-Bacteria Model')
    parser.add_argument('--vb_train', default=False, required=False, action='store_true')

    parser.add_argument('--ph_out', default=[180, 260, 320], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Pneumonia-Healthy Model')
    parser.add_argument('--ph_kernels', default=[5, 4, 3], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Pneumonia-Healthy Model')
    parser.add_argument('--ph_strides', default=[2, 2, 1], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Pneumonia-Healthy Model')
    parser.add_argument('--ph_pooling_details', default=[(2, 2), (2, 2), (2, 1)], required=False, type=tuple, nargs='+',
                        help='Specifies kernel dimensions of Pneumonia-Healthy Model')
    parser.add_argument('--ph_bn_apply', default=[True, True, False], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Pneumonia-Healthy Model')
    parser.add_argument('--ph_drop_cnn', default=[0.2, 0.3, 0.5], required=False, type=float, nargs='+',
                        help='Specifies kernel dimensions of Pneumonia-Healthy Model')
    parser.add_argument('--ph_lin_out', default=[120], required=False, type=int, nargs='+',
                        help='Specifies kernel dimensions of Pneumonia-Healthy Model')
    parser.add_argument('--ph_drop_lin', default=[0.3], required=False, type=float, nargs='+',
                        help='Specifies kernel dimensions of Pneumonia-Healthy Model')
    parser.add_argument('--ph_bn_lin_apply', default=[False], required=False, type=int, nargs='+',
                        help='Specifies batch normalization application for FCN of Virus-Bacteria Model')
    parser.add_argument('--ph_train', default=False, required=False, action='store_true')

    parser.add_argument('--exp_num', required=False, default=13)
    parser.add_argument('--image_dim', default=128, required=False, type=int,
                        help='Specifies desired image size for the model')
    parser.add_argument('--uniform', default=False, required=False, action='store_true',
                        help='Specifies uniform distribution for data will be satisfied or not')

    parser.add_argument('--optimizer', required=False, default='SGD', type=str,
                        help='Specifies optimizer choice: can either be Adam or SGD')
    parser.add_argument('--epochs_om', required=False, default=2, type=int,
                        help='Specifies number of epochs for om model')
    parser.add_argument('--epochs_ph', required=False, default=2, type=int,
                        help='Specifies number of epochs for ph model')
    parser.add_argument('--epochs_vb', required=False, default=2, type=int,
                        help='Specifies number of epochs for vb model')

    parser.add_argument('--resume_training_om', required=False, default=False, action='store_true',
                        help='Specifies whether user wants to resume training for om model or not')
    parser.add_argument('--resume_training_ph', required=False, default=False, action='store_true',
                        help='Specifies whether user wants to resume training for ph model or not')
    parser.add_argument('--resume_training_vb', required=False, default=False, action='store_true',
                        help='Specifies whether user wants to resume training for vb model or not')
    parser.add_argument('--om_num_classes', required=False, default=3, type=int,
                        help='Specifies number of classes for om model')
    parser.add_argument('--ph_num_classes', required=False, default=2, type=int,
                        help='Specifies number of classes for ph model')
    parser.add_argument('--vb_num_classes', required=False, default=2, type=int,
                        help='Specifies number of classes for vb model')
    parser.add_argument('--zero_center', action='store_true', required=False)

    return parser.parse_args()


def get_parameters() -> dict:
    """
    Function collects user-specific data and add device choice (it can change according to machine) and provides
    all required parameters
    :return: dictionary which contains all required parameters for the project
    """
    arguments = set_configuration()
    parameters = dict()
    for argument in vars(arguments):
        parameters[argument] = getattr(arguments, argument)
    parameters['device'] = 'cpu' if not torch.cuda.is_available() else 'cuda'
    return parameters
