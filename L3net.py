import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
'''From Cheng et. al.: Graph Convolution with Low-rank Learnable Local Filters '''

# TODO: change the shared bases version to having generic model, where we just fill columns/rows of B_k by an identity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConv_Bases(nn.Module):
    def __init__(self, in_channels, out_channels, A_, order_list=[1], bias=True):
        """
        A_ is the adjacency matrix, whose power gives us d-hop neighbors
        """
        super().__init__()
        A_.fill_diagonal_(1.)  # Always consider self loops
        self.A_ = A_
        # bases hyper-parameter
        self.num_bases = len(order_list)
        self.order_list = order_list
        # define channel-mixing operation, which is from in_channel -> out_channel
        self.coeff_conv = nn.Conv1d(
            in_channels=self.num_bases*in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False)
        # bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            n = in_channels * self.num_bases
            stdv = 1. / math.sqrt(n)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        # TODO: NOTE: for computational efficiency, we can actually pass in a list of A_ which are "powered", so that we need not compute the matrix_powers on the fly
        self.get_bases()

    def k_th_order_A(self, order):
        """
        modify A to incorporate the right order of neighbors
        :param: A
        :return: new A
        """
        if order == 0:
            return torch.eye(self.A_.shape[1]).float().to(device)
        A_total = torch.zeros_like(self.A_)
        for i in range(1, order + 1):
            A_total += self.A_.matrix_power(i)
        return (A_total != 0).float()

    def get_bases_template(self):
        """
        Get bases_template from self.A_, which has diagonals being 1
        """
        bases_template = []

        for order in self.order_list:
            # assert not (0 in self.order_list)
            bases_template += [self.k_th_order_A(order)]

        bases_template = torch.stack(bases_template, dim=0)
        bases_template.requires_grad = False
        self.bases_template = bases_template

    def get_bases(self):
        """
        Get both bases_template and mask
        """
        # get bases_template from A' and order_list, with shape [num_bases, V, V]
        self.get_bases_template()

        # create bases_mask, with shape [num_bases, V, V]
        self.bases_mask = nn.Parameter(
            torch.Tensor(*(self.bases_template.shape)))
        # init bases, 3: avg. support size
        in_size = self.num_bases * 3 * self.bases_template.shape[0]
        std_ = math.sqrt(1. / in_size)
        nn.init.normal_(self.bases_mask, std=std_)

    def forward(self, input):
        N, in_channels, num_nodes = input.shape
        # first step in dcf
        features_bases = []
        rec_kernel = self.bases_template * self.bases_mask
        for kernel in rec_kernel:
            # each with shape [N, in_channels, num_nodes]
            features_bases += [torch.matmul(input, kernel)]
        # with shape [N, in_channels*num_bases, num_nodes]
        features_bases = torch.cat(features_bases, dim=1)

        # second step, with shape [N, out_channels, num_nodes]
        features_bases = self.coeff_conv(features_bases)

        # add bias
        features_bases += self.bias.unsqueeze(-1)

        return features_bases


# class GraphConv_Bases_Shared(nn.Module):
#
#     def __init__(self, in_channels, out_channels, num_nodes, bias=True,
#                  order_list=[1], num_bases=1):
#         super(GraphConv_Bases_Shared, self).__init__()
#
#         self.num_bases = num_bases
#         self.order_list = order_list
#
#         self.bases = nn.Parameter(torch.Tensor(3))
#         self.num_nodes = num_nodes
#         in_size = 3 * self.num_nodes * 1.0
#         std_ = math.sqrt(1. / in_size)
#         nn.init.normal_(self.bases, std=std_)
#
#         # define coeff operation,
#         # for three types of GCN, this is the same
#         self.coeff_conv = nn.Conv1d(
#             in_channels=self.num_bases*in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             bias=False,
#         )
#
#         # bias
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#             n = in_channels * self.num_bases
#             stdv = 1. / math.sqrt(n)
#             self.bias.data.uniform_(-stdv, stdv)
#         else:
#             self.register_parameter('bias', None)
#
#     def build_local_filter(self):
#         # from [3] to tri-diagonal matrix
#         bases_matrix = []
#         for i in range(self.num_nodes):
#             if i == 0:
#                 bases_matrix.append(
#                     F.pad(self.bases[1:], (0, self.num_nodes-2)))
#             elif i == self.num_nodes-1:
#                 bases_matrix.append(
#                     F.pad(self.bases[:-1], (self.num_nodes-2, 0)))
#             else:
#                 bases_matrix.append(
#                     F.pad(self.bases, (i-1, self.num_nodes-2-i)))
#
#         """
#             Remember: each bases is a column vector of whole matrix
#         """
#         # pdb.set_trace()
#         bases_matrix = torch.stack(bases_matrix, dim=1)
#         return bases_matrix
#
#     def forward(self, input):
#         N, in_channels, num_nodes = input.shape
#         # first step in dcf
#         bases_matrix = self.build_local_filter()
#         features_bases = torch.matmul(input, bases_matrix)
#
#         # second step, with shape [N, out_channels, num_nodes]
#         features_bases = self.coeff_conv(features_bases)
#
#         # add bias
#         features_bases += self.bias.unsqueeze(-1)
#
#         return features_bases


#####
#####
