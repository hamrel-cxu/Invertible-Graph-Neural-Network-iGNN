# from torch.nn.utils.parametrizations import spectral_norm
import FrEIA.modules as Fm
import FrEIA.framework as Ff
from scipy.stats import norm
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd.functional import jacobian
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv
from torch_geometric.nn import Sequential as Graph_Sequential
from torch.nn import Parameter
from L3net import GraphConv_Bases
import torch_geometric as pyg
import ResFlow_logdet as reslogdet
import torch.nn.functional as F
# print(torch.__file__)
import pdb
from timeit import default_timer as timer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# IResNet


class input_tranpose(nn.Module):
    def __init__(self, dim0=1, dim1=2):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class InvResBlock(nn.Module):
    '''
    First construct ResNet block as F_b=I+g, where
        g=W_i \circ \phi ... \circ W_1, where the spectral norm of W_i is strictly smaller than 1 & phi can be chosen as ReLU, ELU, tanh, etc. as contractive nonlinearities
    '''

    def __init__(self, C, model_args, version, A_, act_type='swish'):
        super().__init__()
        '''
        model_args=[c,dim]
            c: The spectral norm of the weight matrix. Too small choice seem to make things harder to train
            dim: Dimension of hidden representation
        '''
        dim, input_non_linear = model_args
        if act_type == 'ELU':
            act = nn.ELU(inplace=True)
        if act_type == 'swish':
            act = Swish()  # NOTE, some original results before submission used ELU, so there may be key mismatch
        layers = layers_append([], act, C, dim, C, version=version, A_=A_)
        self.bottleneck_block = nn.Sequential(*layers)
        self.actnorm = None
        self.logdet = False
        self.C = C

    def forward(self, x):
        n = int(x.shape[1] / self.C)
        x = x.reshape(x.shape[0], n, self.C)
        if self.logdet == False:
            # Exact logdet later
            Fx = self.bottleneck_block(x)
            logdet_block = None  # placeholder
            # The goal is so that the jacobian has shape nC-by-nC
        else:
            Fx, logdet_block = reslogdet.resflow_logdet(self, x)
        # output = Fx/self.delta_t + x
        output = Fx + x
        output = output.flatten(start_dim=1)  # So it is in R^N times nC
        return [output, Fx, logdet_block]

    def inverse(self, y, maxIter=100, eps=1e-7):
        x_pre = y
        for iter in range(maxIter):
            # x_now = y - self.bottleneck_block(x_pre)/self.delta_t
            x_now = y - self.bottleneck_block(x_pre)
            diff = torch.linalg.norm(x_now - x_pre)
            if diff <= eps:
                break
            x_pre = x_now
        if self.actnorm is not None:
            x_now = self.actnorm.inverse(x_now)
        return x_now


class InvResBlock_Graph(nn.Module):
    '''
    First construct ResNet block as F_b=I+g, where
        g=W_tori \circ \phi ... \circ W_1, where the spectral norm of W_i is strictly smaller than 1 & phi can be chosen as ReLU, ELU, tanh, etc. as contractive nonlinearities
    '''

    def __init__(self, C, model_args, version='two_GCN_one_FC', act_type='swish'):
        super().__init__()
        [dim, K] = model_args
        if act_type == 'ELU':
            act = nn.ELU(inplace=True)
        if act_type == 'swish':
            act = Swish()  # NOTE, some original results before submission used ELU, so there may be key mismatch
        # trans = input_tranpose(1, 2)
        layers = []
        layers = layers_append(layers, act, C, dim, C, K, version)
        self.bottleneck_block = Graph_Sequential(
            'x, edge_index, edge_weight', layers)
        # self.actnorm = ActNorm2D(C) # extremely slow
        self.actnorm = None
        self.logdet = False
        self.C = C

    def forward(self, x, edge_index, edge_weight):
        # HERE, need to reshape x first, as it is a flattened matrix of node features (so in dimension nC rather than n-X-C)
        # We assume each row is this flattened graph, and x\in \R^{N\times nC}
        n = int(x.shape[1] / self.C)
        x = x.reshape(x.shape[0], n, self.C)
        if self.logdet == False:
            # Exact logdet later
            Fx = self.bottleneck_block(x, edge_index, edge_weight)
            logdet_block = None  # placeholder
            # The goal is so that the jacobian has shape nC-by-nC
        else:
            Fx, logdet_block = reslogdet.resflow_logdet(
                self, x, edge_index, edge_weight)
        # output = Fx/self.delta_t + x
        output = Fx + x
        output = output.flatten(start_dim=1)  # So it is in R^N times nC
        return [output, Fx, logdet_block]

    def inverse(self, y, edge_index, edge_weight, maxIter=100, eps=1e-7):
        # Fixed point iteration to find the inverse
        x_pre = y
        for iter in range(maxIter):
            # x_now = y - \
            #     self.bottleneck_block(
            #         x_pre, edge_index, edge_weight)/self.delta_t
            x_now = y - self.bottleneck_block(x_pre, edge_index, edge_weight)
            diff = torch.linalg.norm(x_now - x_pre)
            if diff <= eps:
                break
            x_pre = x_now
        if self.actnorm is not None:
            x_now = self.actnorm.inverse(x_now)
        return x_now


class InvResNet(nn.Module):
    '''
    Refer to https://github.com/jhjacobsen/invertible-resnet/blob/master/models/conv_iResNet.py, where they stack multiple blocks together
        Line 418 class conv_iResNet(nn.Module) stacks blocks together,
        Line 56 builds the block
    '''

    def __init__(self, C, output_dim=1, nblocks=5, model_args=[0.9, 64, 3], graph=False, version='two_GCN_one_FC', A_=None, act_type='swish'):
        '''
            Output_dim: for classification
            num_nodes: number of graph nodes
        '''
        super().__init__()
        self.C = C
        dim, K = model_args[1], model_args[2]
        # Actual dimension in which the distribution flows
        self.C = C
        if graph:
            self.blocks = nn.ModuleList([InvResBlock_Graph(
                self.C, [dim, K], version, act_type) for b in range(nblocks)])
        else:
            self.blocks = nn.ModuleList([InvResBlock(
                self.C, [dim, b], version, A_, act_type) for b in range(nblocks)])
        self.fc = nn.Linear(self.C, output_dim)
        if self.C > 2:
            # Only consider 2d feature for now, and the input is thus flattened
            self.fc = nn.Linear(2, output_dim)
        self.reduce_factor = model_args[0]
        self.small_weights()

    def forward(self, x, edge_index=None, edge_weight=None, logdet=True):
        # X is flattened to have the correct Jacobian, so this is V*C
        in_dim = x.shape[1]
        log_det = 0
        transport_cost = 0
        for j, block in enumerate(self.blocks):
            # If store logdet, use residual flow estimation
            block.logdet = logdet if in_dim > 2 else False
            block.delta_t = len(self.blocks)
            x_for, Fx, det_block = block(
                x, edge_index, edge_weight) if edge_index is not None else block(x)
            if logdet:
                if det_block is None:
                    # Over-ride with brute force det
                    if edge_index is not None:
                        det_block = torch.log(
                            torch.abs(torch.det(batch_jacobian(block, x, edge_index, edge_weight))))
                    else:
                        det_block = torch.log(
                            torch.abs(torch.det(batch_jacobian(block, x))))
                log_det = log_det + det_block.sum()
            transport_cost += (torch.linalg.norm(Fx.flatten(start_dim=1),
                               dim=1)**2 / 2).sum()
            x = x_for
        if logdet:
            return x, log_det, transport_cost
        else:
            return x

    def inverse(self, y, edge_index=None, edge_weight=None, maxIter=50):
        with torch.no_grad():
            for block in reversed(self.blocks):
                y = block.inverse(
                    y, edge_index, edge_weight, maxIter) if edge_index is not None else block.inverse(y, maxIter)
        return y

    def classification(self, H):
        '''
            Yield a linear classifier
        '''
        return self.fc(H)

    def small_weights(self):
        for name, W in self.named_parameters():
            if 'fc' not in name:
                with torch.no_grad():
                    # Of course, this is user-specified. It is just for initialization
                    # In fact, should not be too small, as it would make the transport cost too negligible
                    # And losses decay too slowly
                    # And the model more likely get non-invertible...
                    W.mul_(self.reduce_factor)
                W.requires_grad = True
# Small nets


class SmallGenNet(nn.Module):
    '''
        Yield the conditional mean of the base distribution using one-hot encoded response Y
    '''

    def __init__(self, Y_dim, C):
        super().__init__()
        self.fc = nn.Linear(Y_dim, C, bias=False)
        # Initialize "pair-wise far enough mean vectors"
        # Below is naive (and possibly not ideal b/c some are too far), as we just keep "adding" 2*\phi^{-1}(1-\eps) to each component
        # NOTE, for experiments I tried, I used "hand-picked" initialization with X|Y
        delta = 2*norm().ppf(0.99)  # 99% quantile of standard normal
        mu_mat0 = torch.zeros(C, Y_dim)
        for i in range(1, Y_dim):
            mu_mat0[:, i] = mu_mat0[:, i-1]+delta
        with torch.no_grad():
            self.fc.weight.copy_(mu_mat0)

    def forward(self, Y):
        return self.fc(Y)


# Log det


# Brute force
def batch_jacobian(func, x, edge_index=None, edge_weight=None):
    # Basically apply the jacobian function on each sample in the batch
    # x in shape (Batch, Length)
    def _func_sum(x):
        if edge_index is not None:
            return func(x, edge_index, edge_weight)[0].sum(dim=0)
        else:
            return func(x)[0].sum(dim=0)
    return jacobian(_func_sum, x, create_graph=True).permute(1, 0, 2)

# CGAN


class CGAN_net(nn.Module):
    '''
        Note, this is very similar to our IResBlock, but just we no longer concatenate multiple blocks
    '''

    def __init__(self, C, dim, Y_dim=2, nblocks=10, classify=False, graph=True, version='two_L3_two_FC', A_=None):
        super().__init__()
        act = nn.ELU(inplace=True)
        full_layers = []
        trans = input_tranpose(1, 2)
        if nblocks > 1:
            for i in range(nblocks - 1):
                if i == 0:
                    full_layers += layers_append([], act,
                                                 C, dim, dim, version=version, A_=A_)
                else:
                    full_layers += layers_append([], act,
                                                 dim, dim, dim, version=version, A_=A_)
                full_layers.append(trans)
                full_layers.append(pyg.nn.BatchNorm(dim))
                full_layers.append(trans)
            full_layers += layers_append([], act, dim,
                                         dim, C - Y_dim, version=version, A_=A_)
        else:
            full_layers = layers_append([], act, C,
                                        dim, C - Y_dim, version=version, A_=A_)
        self.graph = graph
        if self.graph:
            # e.g., ChebNet
            self.bottleneck_block = Graph_Sequential(
                'x, edge_index, edge_weight', full_layers)
        else:
            self.bottleneck_block = nn.Sequential(*full_layers)
        # self.actnorm = ActNorm2D(C) # extremely slow
        self.actnorm = None
        self.classify = classify
        if self.classify:
            last_layer = layers_append([], act,
                                       C - Y_dim, 32, 1, version='three_FC')
            self.D_output = nn.Sequential(*last_layer)

    def forward(self, x, edge_index, edge_weight):
        if self.graph:
            output = self.bottleneck_block(x, edge_index, edge_weight)
        else:
            output = self.bottleneck_block(x)
        if self.classify:
            # For the min-max GAN
            offset = 1e-4
            output = torch.nn.Sigmoid()(self.D_output(output))
            offset_vec = torch.zeros(output.size()).to(device)
            if (output < offset).sum() > 0:
                offset_vec[output < offset] = (
                    offset - output[output < offset]).clone().detach()
            if (output > 1 - offset).sum() > 0:
                offset_vec[output > 1
                           - offset] = -(output[output > 1 - offset] - (1 - offset)).clone().detach()
            return output + offset_vec
            # # For Wasserstain GAN
            # return self.D_output(output)
        else:
            return output

    def set_requires_grad(self, TorF):
        for param in self.parameters():
            param.requires_grad = TorF


# CINN_Nflow


class cINN_Nflow(nn.Module):
    '''cINN for class-conditional MNISt generation'''

    def __init__(self, in_dim, cond_dim, dim, nblocks, clamp_val):
        super().__init__()
        self.in_dim, self.cond_dim = in_dim, cond_dim
        self.dim, self.nblocks, self.clamp_val = dim, nblocks, clamp_val
        self.cinn = self.build_inn()

    def build_inn(self):
        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, self.dim), nn.ReLU(),
                                 nn.Linear(self.dim,  c_out))

        cond = Ff.ConditionNode(self.cond_dim)
        nodes = [Ff.InputNode(self.in_dim)]

        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(self.nblocks):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet_fc,
                                     'clamp': self.clamp_val},
                                 conditions=cond))
        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, eY):
        z, jac = self.cinn(x, c=eY, jac=True)
        return z, jac

    def reverse_sample(self, z, eY):
        return self.cinn(z, c=eY, rev=True)

# Append net:


def layers_append(layers, act, C, dim, C1, K=3, version='one_Cheb_two_FC', A_=None):
    GCN_layer = GCNConv(C, dim, cached=True)
    if version == 'one_GCN_one_FC':
        layers.append((GCN_layer, 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, C1))
    if version == 'one_GCN_two_FC':
        layers.append((GCN_layer, 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, dim))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, C1))
    if version == 'one_GCN_three_FC':
        layers.append((GCN_layer, 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, dim))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, dim))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, C1))
    if version == 'two_GCN_one_FC':
        layers.append((GCN_layer, 'x, edge_index, edge_weight -> x'))
        # layers.append(pyg.nn.BatchNorm(dim)) # Some issues existed, IDK why
        layers.append(act)
        layers.append(
            (GCN_layer, 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, C1))
    if version == 'one_Cheb_one_FC':
        layers.append(
            (ChebConv(C, dim, K=K), 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, C1))
    if version == 'one_Cheb_two_FC':
        layers.append(
            (ChebConv(C, dim, K=K), 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, dim))
        # # NOTE, Batchnorm makes invertibility somehow not hold, but transpose etc. works
        # layers.append(trans)
        # layers.append(pyg.nn.BatchNorm(dim))
        # layers.append(trans)
        layers.append(act)
        layers.append(torch.nn.Linear(dim, C1))
    if version == 'one_Cheb_three_FC':
        layers.append(
            (ChebConv(C, dim, K=K), 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, dim))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, dim))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, C1))
    if version == 'two_Cheb_two_FC':
        layers.append(
            (ChebConv(C, dim, K=K), 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(
            (ChebConv(dim, dim, K=K), 'x, edge_index, edge_weight -> x'))
        layers.append(act)
        layers.append(torch.nn.Linear(dim, dim))
        # # NOTE, Batchnorm makes invertibility somehow not hold, but transpose etc. works
        # layers.append(trans)
        # layers.append(pyg.nn.BatchNorm(dim))
        # layers.append(trans)
        layers.append(act)
        layers.append(torch.nn.Linear(dim, C1))
    if version == 'one_Cheb':
        layers.append(
            (ChebConv(C, dim, K=K), 'x, edge_index, edge_weight -> x'))
        layers.append(act)
    if version == 'three_FC':
        layers.append(nn.Linear(C, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, C1))
    if version == 'four_FC':
        layers.append(nn.Linear(C, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, C1))
    # For L3net:
    trans = input_tranpose(1, 2)
    order_list = [1]
    if A_ is not None:
        if A_.shape[0] == 3:
            # The simulation example
            order_list = [0, 1, 2]
        if A_.shape[0] == 10:
            # Solar data
            order_list = [1, 2]  # Two bases, with 1 & 2 hop neighbors
        if A_.shape[0] == 20 or A_.shape[0] == 15:
            # Traffic data
            # order_list = [1, 2, 2]  # Two bases, with 1 & 2 hop neighbors
            order_list = [0, 1, 2]
    if version == 'one_L3_two_FC':
        layers.append(trans)
        layers.append(GraphConv_Bases(C, dim, A_, order_list=order_list))
        layers.append(trans)
        layers.append(act)
        layers.append(nn.Linear(dim, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, C1))
    if version == 'one_L3_three_FC':
        layers.append(trans)
        layers.append(GraphConv_Bases(C, dim, A_, order_list=order_list))
        layers.append(trans)
        layers.append(act)
        layers.append(nn.Linear(dim, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, C1))
    if version == 'two_L3_two_FC':
        layers.append(trans)
        layers.append(GraphConv_Bases(C, dim, A_, order_list=order_list))
        layers.append(GraphConv_Bases(dim, dim, A_, order_list=order_list))
        layers.append(trans)
        layers.append(act)
        layers.append(nn.Linear(dim, dim))
        layers.append(act)
        layers.append(nn.Linear(dim, C1))
    if version == 'one_L3':
        layers.append(trans)
        order_list = [0, 1, 2]
        layers.append(GraphConv_Bases(C, dim, A_, order_list=order_list))
        layers.append(trans)
        layers.append(act)
    return layers


#######

#######
