import torch
from torch import nn


class BuildModelStructure:
    """
    Class is used to build network according to model name. It adds layers dynamically according to user's input
    """
    def __init__(self, config_parameters, model_name):
        """
        Method is used as initializer of the class
        :param config_parameters: Configuration parameters of the project
        :param model_name: specifies which model is created
        """
        self.model_name = model_name
        self.configuration = config_parameters
        self.model_base = self.create_model_layers()
        self.num_cnn = len(self.configuration[f'{self.model_name}_out'])
        self.num_lin = len(self.configuration[f'{self.model_name}_lin_out'])

    def create_cnn_model(self) -> dict:
        """
        Method is used for adding layers to Convolutional Neural Network, dynamically
        :return: dictionary, in which keys are layer names and values are layers
        """
        specific_parameters = ['out', 'pooling_details', 'strides', "bn_apply", "drop_cnn"]
        for each in specific_parameters:
            if len(self.configuration[f"{self.model_name}_{each}"]) != len(
                    self.configuration[f"{self.model_name}_kernels"]):
                raise AssertionError(f'Number of {each} for {self.model_name} does not match requirements! All cnn '
                                     f'parameters must be in the same length')
        cnn_model = dict()
        for idx, out_channels in enumerate(self.configuration[f'{self.model_name}_out']):
            if idx == 0:
                cnn_layer = nn.Conv2d(in_channels=1, out_channels=out_channels,
                                      kernel_size=self.configuration[f'{self.model_name}_kernels'][idx],
                                      stride=self.configuration[f'{self.model_name}_strides'][idx])
            else:
                cnn_layer = nn.Conv2d(in_channels=self.configuration[f"{self.model_name}_out"][idx - 1],
                                      out_channels=out_channels,
                                      kernel_size=self.configuration[f'{self.model_name}_kernels'][idx],
                                      stride=self.configuration[f'{self.model_name}_strides'][idx])

            cnn_model[f'cnn_{idx}'] = cnn_layer
            if self.configuration[f'{self.model_name}_bn_apply'][idx]:
                cnn_model[f'bn_{idx}'] = nn.BatchNorm2d(out_channels)
            pooling_details = self.configuration[f'{self.model_name}_pooling_details'][idx]
            cnn_model[f'pooling_{idx}'] = nn.MaxPool2d(pooling_details[0], pooling_details[1])
            cnn_model['drop_cnn_{idx}'] = nn.Dropout(self.configuration[f'{self.model_name}_drop_cnn'][idx])
            cnn_model[f'relu_cnn_{idx}'] = nn.ReLU()
        return cnn_model

    def compute_cnn_layer_out(self) -> int:
        """
        Method is used for computing output dimension of CNN network. It varies because of chosen kernel and stride
        sizes.
        :return: integer specifies output dimension of CNN
        """
        kernels = self.configuration[f'{self.model_name}_kernels']
        pooling_details = self.configuration[f'{self.model_name}_pooling_details']
        strides = self.configuration[f'{self.model_name}_strides']
        out = self.configuration['image_dim']
        for each in range(len(kernels)):
            out = ((out - kernels[each]) / strides[each]) + 1
            out = ((out - pooling_details[each][0]) / pooling_details[each][1]) + 1

        return int(out)

    def create_linear_layers(self) -> dict:
        """
        Method is used for adding layers to Fully Connected Network, dynamically,  which will be added on top of the CNN
        :return: dictionary, in which keys are layer names and values are layers
        """
        for each in ['drop_lin', 'bn_lin_apply']:
            if len(self.configuration[f'{self.model_name}_lin_out']) != len(
                    self.configuration[f'{self.model_name}_{each}']):
                raise AssertionError(f'Number of linear layers and dropout layers for {self.model_name} does not match '
                                     f'requirements! All cnn parameters must be in the same length. If you do not want '
                                     f'to use dropout specify corresponding dropout layer with 0')
        linear_model = dict()
        cnn_out = self.compute_cnn_layer_out()
        for idx, lin_out in enumerate(self.configuration[f'{self.model_name}_lin_out']):
            if idx == 0:
                linear = nn.Linear(self.configuration[f'{self.model_name}_out'][-1] * cnn_out * cnn_out, lin_out)
            else:
                linear = nn.Linear(self.configuration[f'{self.model_name}_lin_out'][idx - 1], lin_out)

            linear_model[f'linear_{idx}'] = linear
            if self.configuration[f'{self.model_name}_bn_lin_apply'][idx]:
                linear_model[f'linear_bn_{idx}'] = nn.BatchNorm1d(lin_out)
            linear_model[f'dropout_{idx}'] = nn.Dropout(self.configuration[f'{self.model_name}_drop_lin'][idx])
            linear_model[f'relu_{idx}'] = nn.ReLU()
        linear_model['linear_out'] = nn.Linear(self.configuration[f'{self.model_name}_lin_out'][-1],
                                               self.configuration[f'{self.model_name}_num_classes'])
        linear_model[f'relu_out'] = nn.ReLU()
        return linear_model

    def create_model_layers(self) -> dict:
        """
        Method combines all layers in one dictionary so that we will use it as our model
        :return: dictionary, in which keys are layer names and values are layers
        """
        model_layers = self.create_cnn_model()
        for name, layer in self.create_linear_layers().items():
            model_layers[name] = layer
        return model_layers

    def __iter__(self):
        """
        Method is used as default iterator of the class to iterate over layers of the object
        :yield: tuple of name of layer and layer itself
        """
        for name, module in self.model_base.items():
            yield name, module


class CNNModel(nn.Module):
    """
    Class is an object for classifier model
    """
    def __init__(self, model_structure_object: BuildModelStructure):
        """
        Method is used as initializer of the class
        :param model_structure_object: model structure object in which all layers are added, dynamically
        """
        super(CNNModel, self).__init__()
        self.model_structure_object = model_structure_object
        for name, module in model_structure_object:
            self.add_module(name, module)

    def forward(self, input_data: torch.FloatTensor) -> torch.FloatTensor:
        """
        Method is used as feedforward step of the network
        :param input_data: input data which can be single or in batches
        :return: output of the model
        """
        cnn_counter = 0
        for module in self.children():
            if type(module) == nn.Conv2d:
                cnn_counter += 1
            if cnn_counter == self.model_structure_object.num_cnn and type(module) == nn.Linear:
                input_data = input_data.view(input_data.size(0), -1)
                input_data = module(input_data)
                cnn_counter = 0
            else:
                input_data = module(input_data)
        return input_data
