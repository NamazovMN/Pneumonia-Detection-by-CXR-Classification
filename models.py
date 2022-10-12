import torch.nn as nn
from utilities import *


class CNN_model_om(nn.Module):
    def __init__(self, model_parameters):
        super(CNN_model_om, self).__init__()
        self.model_parameters = model_parameters
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv2d(
            in_channels=self.model_parameters['in_size'],
            out_channels=self.model_parameters['out1'],
            kernel_size=self.model_parameters['kernel1'],
            stride=self.model_parameters['stride1']
        )
        self.cnn2 = nn.Conv2d(
            in_channels=self.model_parameters['out1'],
            out_channels=self.model_parameters['out2'],
            kernel_size=self.model_parameters['kernel2'],
            stride=self.model_parameters['stride2']
        )
        self.cnn3 = nn.Conv2d(
            in_channels=self.model_parameters['out2'],
            out_channels=self.model_parameters['out3'],
            kernel_size=self.model_parameters['kernel3'],
            stride=self.model_parameters['stride3']
        )
        #
        # self.cnn4 = nn.Conv2d(
        #     in_channels=self.model_parameters['out3'],
        #     out_channels=self.model_parameters['out4'],
        #     kernel_size=self.model_parameters['kernel4'],
        #     stride=self.model_parameters['stride4']
        # )

        out_cnn = compute_output(self.model_parameters, 3)
        print(f'Output of Convolutional Layers is in OM : {out_cnn}')

        self.linear1 = nn.Linear(in_features=self.model_parameters['out3'] * out_cnn * out_cnn,
                                 out_features=self.model_parameters['linout1'])
        self.linear2 = nn.Linear(in_features=self.model_parameters['linout1'],
                                 out_features=self.model_parameters['linout2'])
        # self.linear3 = nn.Linear(in_features=self.model_parameters['linout2'],
        #                          out_features=self.model_parameters['linout3'])

        self.out = nn.Linear(in_features=self.model_parameters['linout2'],
                             out_features=self.model_parameters['out_size'])

        self.mp1 = nn.MaxPool2d(self.model_parameters['mp_kernel1'], self.model_parameters['mp_stride1'])
        self.mp2 = nn.MaxPool2d(self.model_parameters['mp_kernel2'], self.model_parameters['mp_stride2'])
        self.mp3 = nn.MaxPool2d(self.model_parameters['mp_kernel3'], self.model_parameters['mp_stride3'])
        # self.mp4 = nn.MaxPool2d(self.model_parameters['mp_kernel4'], self.model_parameters['mp_stride4'])

        self.bn1 = nn.BatchNorm2d(self.model_parameters['out1'])
        self.bn2 = nn.BatchNorm2d(self.model_parameters['out2'])
        self.bn3 = nn.BatchNorm2d(self.model_parameters['out3'])
        # self.bn4 = nn.BatchNorm2d(self.model_parameters['out4'])

        self.lin_bn1 = nn.BatchNorm1d(self.model_parameters['linout1'])
        self.lin_bn2 = nn.BatchNorm1d(self.model_parameters['linout2'])

        self.dp1 = nn.Dropout(self.model_parameters['dp1'])
        self.dp2 = nn.Dropout(self.model_parameters['dp2'])
        self.dp3 = nn.Dropout(self.model_parameters['dp3'])
        self.linear_dp = nn.Dropout(0.4)

    def forward(self, input_data):
        out1 = self.cnn1(input_data)
        bn1 = self.bn1(out1) if self.model_parameters['isbn1'] else out1
        act1 = self.relu(bn1)
        mp1 = self.mp1(act1)

        out2 = self.cnn2(mp1)
        bn2 = self.bn2(out2) if self.model_parameters['isbn2'] else out2
        act2 = self.relu(bn2)
        act2 = self.dp1(act2)
        mp2 = self.mp2(act2)

        out3 = self.cnn3(mp2)
        bn3 = self.bn3(out3) if self.model_parameters['isbn3'] else out3
        act3 = self.relu(bn3)
        act3 = self.dp2(act3)
        mp3 = self.mp3(act3)

        # out4 = self.cnn4(mp3)
        # bn4 = self.bn4(out4) if self.model_parameters['isbn4'] else out4
        # act4 = self.relu(bn4)
        # act4 = self.dp3(act4)
        # mp4 = self.mp4(act4)

        in_lin = mp3.view(mp3.size(0), -1)

        ln1 = self.linear1(in_lin)
        dp1 = self.dp1(ln1)
        # bnlin1 = self.lin_bn1(dp1)
        act4 = self.relu(dp1)

        ln2 = self.linear2(act4)
        dp2 = self.dp2(ln2)
        # bnlin2 = self.lin_bn2(dp2)
        act5 = self.relu(dp2)

        # ln3 = self.linear3(act5)
        # dp3 = self.dp3(ln3)
        # # bnlin2 = self.lin_bn2(dp2)
        # act6 = self.relu(dp3)

        ln3 = self.out(act5)
        # ln3 = nn.Softmax(ln3)
        out = self.relu(ln3)

        return out


class CNN_model_np(nn.Module):
    def __init__(self, model_parameters):
        super(CNN_model_np, self).__init__()
        self.model_parameters = model_parameters
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv2d(
            in_channels=self.model_parameters['in_size'],
            out_channels=self.model_parameters['out1'],
            kernel_size=self.model_parameters['kernel1'],
            stride=self.model_parameters['stride1']
        )
        self.cnn2 = nn.Conv2d(
            in_channels=self.model_parameters['out1'],
            out_channels=self.model_parameters['out2'],
            kernel_size=self.model_parameters['kernel2'],
            stride=self.model_parameters['stride2']
        )

        self.cnn3 = nn.Conv2d(
            in_channels=self.model_parameters['out2'],
            out_channels=self.model_parameters['out3'],
            kernel_size=self.model_parameters['kernel3'],
            stride=self.model_parameters['stride3']
        )

        out_cnn = compute_output(self.model_parameters, 3)
        print(f'Output of Convolutional Layers is in NP : {out_cnn}')

        self.linear1 = nn.Linear(in_features=self.model_parameters['out3'] * out_cnn * out_cnn,
                                 out_features=self.model_parameters['linout1'])
        self.linear2 = nn.Linear(in_features=self.model_parameters['linout1'],
                                 out_features=self.model_parameters['linout2'])
        self.linear3 = nn.Linear(in_features=self.model_parameters['linout2'],
                                 out_features=self.model_parameters['out_size'])

        self.mp1 = nn.MaxPool2d(self.model_parameters['mp_kernel1'], self.model_parameters['mp_stride1'])
        self.mp2 = nn.MaxPool2d(self.model_parameters['mp_kernel2'], self.model_parameters['mp_stride2'])
        self.mp3 = nn.MaxPool2d(self.model_parameters['mp_kernel3'], self.model_parameters['mp_stride3'])

        self.bn1 = nn.BatchNorm2d(self.model_parameters['out1'])
        self.bn2 = nn.BatchNorm2d(self.model_parameters['out2'])
        self.bn3 = nn.BatchNorm2d(self.model_parameters['out3'])

        self.linbn1 = nn.BatchNorm1d(self.model_parameters['linout1'])
        self.linbn2 = nn.BatchNorm1d(self.model_parameters['linout2'])

        self.dp1 = nn.Dropout(self.model_parameters['dp1'])
        self.dp2 = nn.Dropout(self.model_parameters['dp2'])
        self.dp3 = nn.Dropout(self.model_parameters['dp3'])

    def forward(self, input_data):
        out1 = self.cnn1(input_data)
        act1 = self.relu(out1)
        bn1 = self.bn1(act1) if self.model_parameters['isbn1'] else act1
        mp1 = self.mp1(bn1)

        out2 = self.cnn2(mp1)
        act2 = self.relu(out2)
        bn2 = self.bn2(act2) if self.model_parameters['isbn2'] else act2
        bn2 = self.dp2(bn2)
        mp2 = self.mp2(bn2)

        out3 = self.cnn3(mp2)
        act3 = self.relu(out3)
        bn3 = self.bn3(act3) if self.model_parameters['isbn3'] else act3
        bn3 = self.dp3(bn3)
        mp3 = self.mp3(bn3)

        in_lin = mp3.view(mp3.size(0), -1)

        ln1 = self.linear1(in_lin)
        dplin1 = self.dp1(ln1)
        bnlin1 = self.linbn1(dplin1)
        act3 = self.relu(bnlin1)

        ln2 = self.linear2(act3)
        dplin2 = self.dp1(ln2)
        bnlin2 = self.linbn2(dplin2)
        act4 = self.relu(bnlin2)

        ln3 = self.linear3(act4)
        dplin3 = self.dp1(ln3)
        out = self.relu(dplin3)

        return out


class CNN_model_vb(nn.Module):
    def __init__(self, model_parameters):
        super(CNN_model_vb, self).__init__()
        self.model_parameters = model_parameters
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv2d(
            in_channels=self.model_parameters['in_size'],
            out_channels=self.model_parameters['out1'],
            kernel_size=self.model_parameters['kernel1'],
            stride=self.model_parameters['stride1']
        )
        self.cnn2 = nn.Conv2d(
            in_channels=self.model_parameters['out1'],
            out_channels=self.model_parameters['out2'],
            kernel_size=self.model_parameters['kernel2'],
            stride=self.model_parameters['stride2']
        )

        self.cnn3 = nn.Conv2d(
            in_channels=self.model_parameters['out2'],
            out_channels=self.model_parameters['out3'],
            kernel_size=self.model_parameters['kernel3'],
            stride=self.model_parameters['stride3']
        )
        #
        # self.cnn4 = nn.Conv2d(
        #     in_channels=self.model_parameters['out3'],
        #     out_channels=self.model_parameters['out4'],
        #     kernel_size=self.model_parameters['kernel4'],
        #     stride=self.model_parameters['stride4']
        # )

        out_cnn = compute_output(self.model_parameters, 3)
        print(f'Output of Convolutional Layers is in VB: {out_cnn}')

        self.linear1 = nn.Linear(in_features=self.model_parameters['out3'] * out_cnn * out_cnn,
                                 out_features=self.model_parameters['linout1'])
        self.linear2 = nn.Linear(in_features=self.model_parameters['linout1'],
                                 out_features=self.model_parameters['linout2'])
        self.linear3 = nn.Linear(in_features=self.model_parameters['linout2'],
                                 out_features=self.model_parameters['out_size'])

        self.mp1 = nn.MaxPool2d(self.model_parameters['mp_kernel1'], self.model_parameters['mp_stride1'])
        self.mp2 = nn.MaxPool2d(self.model_parameters['mp_kernel2'], self.model_parameters['mp_stride2'])
        self.mp3 = nn.MaxPool2d(self.model_parameters['mp_kernel3'], self.model_parameters['mp_stride3'])
        # self.mp4 = nn.MaxPool2d(self.model_parameters['mp_kernel4'], self.model_parameters['mp_stride4'])

        self.bn1 = nn.BatchNorm2d(self.model_parameters['out1'])
        self.bn2 = nn.BatchNorm2d(self.model_parameters['out2'])
        self.bn3 = nn.BatchNorm2d(self.model_parameters['out3'])
        # self.bn4 = nn.BatchNorm2d(self.model_parameters['out4'])

        self.dp1 = nn.Dropout(self.model_parameters['dp1'])
        self.dp2 = nn.Dropout(self.model_parameters['dp2'])
        self.dp3 = nn.Dropout(self.model_parameters['dp3'])

    def forward(self, input_data):
        out1 = self.cnn1(input_data)
        act1 = self.relu(out1)
        bn1 = self.bn1(act1) if self.model_parameters['isbn1'] else act1
        mp1 = self.mp1(bn1)

        out2 = self.cnn2(mp1)
        act2 = self.relu(out2)
        bn2 = self.bn2(act2) if self.model_parameters['isbn2'] else act2
        mp2 = self.mp2(bn2)

        out3 = self.cnn3(mp2)
        act3 = self.relu(out3)
        bn3 = self.bn3(act3) if self.model_parameters['isbn3'] else act3
        mp3 = self.mp3(bn3)
        #
        # out4 = self.cnn4(mp3)
        # act4 = self.relu(out4)
        # bn4 = self.bn4(act4) if self.model_parameters['isbn4'] else act4
        # mp4 = self.mp3(bn4)

        in_lin = mp3.view(mp3.size(0), -1)
        ln1 = self.linear1(in_lin)
        # dpln1 = self.dp1(ln1)
        act4 = self.relu(ln1)

        ln2 = self.linear2(act4)
        # dpln2 = self.dp2(ln2)
        act5 = self.relu(ln2)

        ln3 = self.linear3(act5)
        out = self.relu(ln3)
        return out
