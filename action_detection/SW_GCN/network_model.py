import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from action_detection.joints import joint_links
from action_detection.SW_GCN.st_gcn_unit import st_gcn
from action_detection.SW_GCN.action_labels import action_labels


def start_cuda():
    """
    set up gpu device to push calculations on
    :return:
    """
    # clear cuda memory when starting execution
    torch.cuda.empty_cache()

    # check if cuda is available
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise Exception("Cuda is not available.")

    device = torch.device("cuda")
    # optimize error messages when calculating on cuda
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    return device


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    This implementation is based on: https://github.com/DelamareMicka/SW-GCN

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """



    def __init__(self, in_channels, num_class, edge_importance_weighting, device, **kwargs):
        super().__init__()

        # load graph:

        self.graph = joint_links  # spatial graph
        A = torch.tensor(self.graph, dtype=torch.float32, requires_grad=False).to(device)  # self in front?
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))  # normalize
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            # ToDo: try different output sizes of stgcn layers
            # ToDo: less layer because few training data

            st_gcn(in_channels, 32, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(32, 32, kernel_size, stride=1, **kwargs),
            st_gcn(32, 32, kernel_size, stride=1, **kwargs),
            st_gcn(32, 32, kernel_size, stride=1, **kwargs),
            st_gcn(32, 64, kernel_size, stride=2, **kwargs),
            st_gcn(64, 64, kernel_size, stride=1, **kwargs),
            st_gcn(64, 64, kernel_size, stride=1, **kwargs),
            st_gcn(64, 128, kernel_size, stride=2, **kwargs),
            st_gcn(128, 128, kernel_size, stride=1, **kwargs),
            st_gcn(128, 128, kernel_size, stride=1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # convolutional
        self.conv_layer = nn.Conv2d(128, 64, (3, 3))

        # dense layer
        self.fcl1 = nn.Linear(128, 64)
        self.fcl2 = nn.Linear(64, 64)
        self.fcl3 = nn.Linear(64, len(action_labels))

        # softmax classifier as last layer
        self.last_layer = nn.Softmax(dim=1)

    # def forward(self, x):
    #
    #     # data normalization
    #     N, C, T, V, M = x.size()  # N:
    #     x = x.permute(0, 4, 3, 1, 2).contiguous()
    #     x = x.view(N * M, V * C, T)
    #     x = self.data_bn(x)
    #     x = x.view(N, M, V, C, T)
    #     x = x.permute(0, 1, 3, 4, 2).contiguous()
    #     x = x.view(N * M, C, T, V)
    #
    #     for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
    #         x, _ = gcn(x, self.A * importance)
    #
    #     # convolutional layer
    #     # x = self.conv_layer(x)
    #
    #     # global pooling
    #     a=x.size()
    #     x = F.avg_pool2d(x, x.size()[2:])
    #     b = x.size()
    #     x = x.view(N, M, -1, 1, 1).mean(dim=1)
    #
    #     # flatten layer
    #     b = x.size()
    #     x = x.view(x.size()[0], -1)
    #
    #     # before softmax after pooling try 2 or 3 fully connected layers for classification
    #     x = self.fcl1(x)
    #     # x = self.fcl2(x)
    #     x = self.fcl3(x)
    #
    #     # prediction
    #     x = self.last_layer(x)
    #     x = x.view(x.size(0), -1)
    #
    #     return x

    def forward(self, x):
        """
        new forward method for leaving out the first dimension in input
        to make network capable of processing live input

        :param x:
        :return:
        """

        # data normalization
        C, T, V, M = x.size()  # N:
        x = x.permute(3, 2, 0, 1).contiguous()
        x = x.view(M, V * C, T)
        x = self.data_bn(x)
        x = x.view(M, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # convolutional layer
        # x = self.conv_layer(x)

        # global pooling
        a = x.size()[2:]
        x = F.avg_pool2d(x, x.size()[2:])
        # x = x.view(M, -1, 1, 1).mean(dim=1)

        # flatten layer
        x = x.view(-1, x.size()[1])

        # before softmax after pooling try 2 or 3 fully connected layers for classification
        x = self.fcl1(x)
        x = self.fcl2(x)
        x = self.fcl3(x)

        # prediction
        x = self.last_layer(x)
        x = x.view(x.size(0), -1)

        return x

    # FixMe: not changed according to dimension change yet (not in use)
    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature
