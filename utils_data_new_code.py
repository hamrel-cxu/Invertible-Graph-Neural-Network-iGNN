import math
from scipy.stats import norm
import matplotlib.cm
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np
import pandas as pd
import torch
import itertools
from sklearn.datasets import make_moons, make_circles
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv
import pdb
import pickle5 as pickle
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' 1. Non_graph Simulation Helpers '''


class graph_simulation_large():
    def __init__(self, A, num_sample_per_Y, num_Y_to_sample):
        '''
            A denotes adjacency matrix
            num_Y_to_sample: Y in [K]^N, so it cannot be sampled for every Y. Thus, just determine how many we want
        '''
        self.num_sample_per_Y = num_sample_per_Y
        self.A = A.float()
        D_inv = torch.diag(1 / A.sum(axis=1))
        P_mat = D_inv @ self.A
        delta = 0.2
        # self.P = (1-delta)*torch.eye(A.shape[0])+delta*(P_mat+P_mat@P_mat)
        self.P = (1-delta)*torch.eye(A.shape[0])+delta*P_mat
        self.N = A.shape[0]
        self.num_Y_to_sample = num_Y_to_sample
        self.get_all_Y()  # Get all Y out of [K]^N choice

    def get_X_Y(self):
        X_full = []
        Y_full = []
        phi = torch.tensor(0.5*math.pi)
        s = torch.sin(phi)
        c = torch.cos(phi)
        rot = torch.stack([torch.stack([c, -s]),
                           torch.stack([s, c])])
        for i in range(self.num_Y_to_sample):
            Y = self.Ys[i]
            self.get_Z_from_Y(Y)
            X = self.P@self.Z
            # X = X@rot
            X_full.append(X.float())
            Y_full.append(Y.repeat(self.num_sample_per_Y, 1).float())
        idx = torch.randperm(self.num_Y_to_sample
                             * self.num_sample_per_Y)  # Randomly shuffle
        X_full, Y_full = torch.vstack(X_full)[idx], torch.vstack(Y_full)[idx]
        self.X_full, self.Y_full = X_full.to(device), Y_full.to(device)
        # # If want to check what are Y and how many of them are there:
        # torch.unique(self.Y_full,dim=0,return_counts=True)

    def get_Z_from_Y(self, Y):
        sigma = 1
        delta = 2*norm().ppf(0.9999)*sigma
        delta = delta*1.5
        mean0, mean1 = torch.tensor([0, delta]).float(), torch.zeros(2).float()
        self.delta = delta
        cov = sigma*torch.eye(2).float()
        base0_dist = MultivariateNormal(mean0, cov)
        base1_dist = MultivariateNormal(mean1, cov)
        # First get Z
        Z = []
        for i, y in enumerate(Y):
            base_dist = base0_dist if y == 0 else base1_dist
            H_sample = base_dist.rsample(sample_shape=(self.num_sample_per_Y,))
            Z.append(H_sample)
        Z = torch.hstack(Z)
        Z = Z.reshape(self.num_sample_per_Y, self.N, 2)
        self.Z = Z

    def get_all_Y(self):
        self.Ys = torch.zeros(self.num_Y_to_sample, self.N)
        for i in range(self.num_Y_to_sample):
            torch.manual_seed(1103+i)
            perm = torch.randperm(self.N)
            idx = perm[:int(self.N/2)]
            self.Ys[i, idx] = 1


class graph_simulate_3node():
    def __init__(self, num_sample):
        self.num_sample = num_sample
        # Control the design of Z
        self.complex_X = False  # If Z are two-moon with rotation + raw H translation
        # Control how Z goes to X, where either "small_averaging" or "disporportional" is used
        self.small_averaging = False  # If A[i,i] += 10
        self.disporportional = False  # If A has different diagonal entries
        self.change_A = False
        self.P_square = False  # If X=P^2Z
        self.plot_X_Z = False

    def get_full_data(self):
        X_full = []
        Y_full = []
        for i, Y in enumerate(itertools.product(*[[0, 1], [0, 1], [0, 1]])):
            print(f'Y={Y}')
            if self.complex_X:
                self.get_Z_from_Y_non_symmetric(Y)
            else:
                self.get_Z_from_Y(Y)
                # X, Z = self.get_X_from_Y(Y)
            self.get_X_from_Z()
            if i == 0:
                print(self.P)
            X_full.append(self.X.float())
            Y_full.append(torch.tensor(Y).repeat(self.num_sample, 1).float())
            if self.plot_X_Z:
                fig, ax = plt.subplots(2, 1, figsize=(8, 4))
                ax[0].scatter(self.Z.flatten(start_dim=0, end_dim=1)[
                    :, 0], self.Z.flatten(start_dim=0, end_dim=1)[:, 1])
                ax[0].set_title('Z')
                ax[1].scatter(self.X.flatten(start_dim=0, end_dim=1)[
                    :, 0], self.X.flatten(start_dim=0, end_dim=1)[:, 1])
                ax[1].set_title('X')
                fig.tight_layout()
                plt.show()
                plt.close()
        torch.manual_seed(1103)
        idx = torch.randperm(self.num_sample * 8)  # Randomly shuffle
        X_full, Y_full = torch.vstack(X_full)[idx], torch.vstack(Y_full)[idx]
        self.X_full, self.Y_full = X_full, Y_full

    def get_Z_from_Y(self, Y):
        # Highly symmetric Z, can cause training issues
        mean0, mean1, cov = torch.tensor([0., 1.5]), torch.tensor(
            [0., -1.5]), (torch.eye(2) * 0.1)
        base0_dist = MultivariateNormal(mean0, cov)
        base1_dist = MultivariateNormal(mean1, cov)
        # First get Z
        Z = []
        for i, y in enumerate(Y):
            base_dist = base0_dist if y == 0 else base1_dist
            H_sample = base_dist.rsample(sample_shape=(self.num_sample,))
            offset = 4
            if i == 0:
                H_sample[:, 0] -= offset
            if i == 2:
                H_sample[:, 0] += offset
            Z.append(H_sample)
        Z = torch.hstack(Z)
        Z = Z.reshape(self.num_sample, 3, 2)
        self.Z = Z

    def get_Z_from_Y_non_symmetric(self, Y):
        # Very non-convex Z, can cause learning difficulties
        mean0, mean1, cov = torch.tensor([0., 1.5]), torch.tensor(
            [0., -1.5]), (torch.eye(2) * 0.1)
        base0_dist = MultivariateNormal(mean0, cov)
        base1_dist = MultivariateNormal(mean1, cov)
        # First get Z
        Z = []
        moon_shift_dict = {0: [0.57, 0.73], 1: [-0.57, -0.73]}
        offset_dict = {0: [-4, 1], 1: [0, 2], 2: [4, 0]}
        theta = np.radians(270)
        c, s = np.cos(theta), np.sin(theta)
        rotate_matrix = np.array(((c, -s), (s, c)))
        for i, y in enumerate(Y):
            # Highly tailored, so we design depending on i and y
            # See notability image for example
            rstate = 1103 + i * 3 + y * 10
            moon_shift, offset = moon_shift_dict[y], offset_dict[i]
            if i == 0:
                X_np, y_np, _, _ = gen_two_moon_data(
                    2 * self.num_sample, 'two_moon', random_state=rstate)
                if y == 0:
                    # Upper moon, with shift
                    Z_sample = X_np[y_np == 0]
                else:
                    Z_sample = rotate_matrix.dot(X_np[y_np == 1].T).T
                Z_sample[:, 0] += moon_shift[0] + offset[0]
                Z_sample[:, 1] += moon_shift[1] + offset[1]
                Z_sample = torch.from_numpy(Z_sample).float()
            elif i == 1:
                if y == 0:
                    base_dist = base0_dist
                    Z_sample = base_dist.rsample(
                        sample_shape=(self.num_sample,))
                    Z_sample[:, 0] += offset[0]
                    Z_sample[:, 1] += offset[1]
                else:
                    X_np, y_np, _, _ = gen_two_moon_data(
                        2 * self.num_sample, 'two_moon', random_state=rstate)
                    Z_sample = X_np[y_np == 1]
                    Z_sample[:, 0] += moon_shift[0] + offset[0]
                    Z_sample[:, 1] += moon_shift[1] + offset[1]
                    Z_sample = torch.from_numpy(Z_sample).float()
            else:
                if y == 0:
                    X_np, y_np, _, _ = gen_two_moon_data(
                        2 * self.num_sample, 'two_moon', random_state=rstate)
                    Z_sample = rotate_matrix.dot(X_np[y_np == 0].T).T
                    Z_sample[:, 0] += moon_shift[0] + offset[0]
                    Z_sample[:, 1] += moon_shift[1] + offset[1]
                    Z_sample = torch.from_numpy(Z_sample).float()
                else:
                    base_dist = base1_dist
                    Z_sample = base_dist.rsample(
                        sample_shape=(self.num_sample,))
                    Z_sample[:, 0] += offset[0]
                    Z_sample[:, 1] += offset[1]
            Z.append(Z_sample)
        Z = torch.hstack(Z)
        Z = Z.reshape(self.num_sample, 3, 2)
        self.Z = Z

    def get_X_from_Z(self):
        # Then get X from Z
        A = np.ones((3, 3))
        A[0, 2], A[2, 0] = 0, 0
        # Emphasize the connection between node 0 and 1, which is used in training.
        if self.change_A:
            A[0, 1] = 2
            A[1, 0] = 2
        if self.small_averaging:
            for i in range(3):
                A[i, i] += 10
        # Disporportional weight
        if self.disporportional:
            A[0, 0] = 2
            A[1, 1] = 3
            A[2, 2] = 5
        D_inv = np.diag(1 / np.sum(A, axis=1))
        P_mat = torch.from_numpy(D_inv.dot(A)).type(torch.float)
        if self.P_square:
            P_mat = P_mat @ P_mat
        self.P = P_mat
        X = P_mat @ self.Z
        self.X = X

    def select_Y(self, Y_rows):
        for i, Y_row in enumerate(Y_rows):
            idx_temp = (self.Y_full == Y_row).all(dim=1).to(device)
            if i == 0:
                idx = idx_temp.clone()
            else:
                idx = torch.logical_or(idx, idx_temp)
        self.X_full, self.Y_full = self.X_full[idx], self.Y_full[idx]


def is_psd(mat):
    eigs = torch.sort(torch.eig(mat)[0][:, 0])[0]
    print(f'Min eigenvalue is {eigs[0]:.2e}')
    print(f'Max eigenvalue is {eigs[-1]:.2e}')
    if bool((mat == mat.T).all() and (eigs >= 0).all()):
        print('matrix is positive semi-definite')
    else:
        raise ValueError('Negative definite matrix from Chebnet')


def quick_sample(X_dist, num_sample, V, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    X = X_dist.rsample(sample_shape=(
        num_sample,)).cpu().detach().reshape(num_sample, V, 1)
    Y = torch.zeros(num_sample, V)
    return X, Y


class GP_graph():
    def __init__(self, num_sample, V):
        self.num_sample = num_sample
        self.V = V  # Must be a prime
        if isPrime(self.V) == False:
            raise ValueError('V Must be a prime for Chordal Cycle Graph')

    def gen_1d_GP_data(self, Sigma_type='ChebNet', perturb_a=True):
        if Sigma_type == 'ChebNet':
            self.get_graph_for_Cheb()
            # So sigma = I + L, normalized and rescaled
            layer = ChebConv(self.V, self.V, K=3).to(device)
            for name, param in layer.named_parameters():
                if name != 'bias':
                    with torch.no_grad():
                        weight_val = torch.eye(self.V).to(device)
                        b = 0
                        c = 0.5
                        a = c  # This is because \hat L^2 has eigenvalue close to 0
                        if perturb_a:
                            a += 0.1
                        # By design, Sigma^{-1} = (a-c)I_v + b*\hat L + 2*c*\hat L^2
                        # Because \hat L has eigenvalue in [-1,1], we need b small to avoid
                        # nan entries when checking psd
                        # Thus, b is just a small number
                        if '0' in name:
                            weight_val *= a
                        if '1' in name:
                            weight_val *= b
                        if '2' in name:
                            weight_val *= c
                        # To make sure covariance matrix invertible
                        param.data = weight_val
            X = torch.eye(self.V, self.V).to(device)
            n_edge = self.edge_index.shape[1]
            edge_weights = torch.ones(n_edge)
            self.edge_weights = edge_weights.to(device)
            # Parametrize Sigma inv
            Sigma_inv = layer(X, self.edge_index,
                              edge_weight=self.edge_weights)
            self.Sigma_inv = Sigma_inv
            Sigma = torch.inverse(Sigma_inv).to(device)
            # Sigma = torch.cholesky_inverse(Sigma_inv).to(device)
            self.Sigma = Sigma
            self.layer = layer
            # Just for numerical consistency
            Sigma = (Sigma+Sigma.T)/2
            is_psd(Sigma)
            gaid = torch.diag(1 / torch.diag(Sigma)**0.5)
            Sigma_corr = gaid @ Sigma @ gaid
            print(f'Corr matrix: {Sigma_corr}')
            Unique_Y, counts_Y = torch.unique(
                torch.round(Sigma_corr, decimals=2), return_counts=True)
            idx = torch.abs(Unique_Y) > 0.1
            print(
                f'Correlation Dist. are {Unique_Y[idx].cpu().detach().tolist()}, \n with frequency {counts_Y[idx].tolist()}')
            cmap = matplotlib.cm.get_cmap('seismic')
            vmin, vmax = -1, 1
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].matshow(torch.eye(self.V, self.V).numpy(),
                          cmap=cmap, vmin=vmin, vmax=vmax)
            c = ax[1].matshow(Sigma_corr.cpu().detach().numpy(),
                              cmap=cmap, vmin=vmin, vmax=vmax)
            cbar_ax = fig.add_axes([1, 0.1, 0.1, 0.8])
            plt.colorbar(c, cax=cbar_ax)
            # plt.colorbar(c, ax=ax.ravel().tolist())
            ax[1].set_title(r'Corr of $\Sigma$')
            ax[0].set_title(r'Identity matrix for ref')

        if Sigma_type == 'Local':
            # Consider KNN graph, where I just manually change the value of correlation matrix Sigma after created
            # NOTE, the 3-node graph has issue that \Sigma^-1 not local
            self.get_graph_and_Sigma_for_local()
            self.edge_weights = torch.ones(self.edge_index.shape[1]).to(device)
            Sigma = self.Sigma
        Mu = torch.zeros(self.V).to(device)
        X_dist = MultivariateNormal(Mu, Sigma)
        seed_train, seed_test = 1103, 111
        seed_train, seed_test = None, None
        self.X_train, self.Y_train = quick_sample(
            X_dist, self.num_sample, self.V, seed=seed_train)
        self.X_test, self.Y_test = quick_sample(
            X_dist, self.num_sample, self.V, seed=seed_test)

    def get_graph_for_Cheb(self):
        # Get a graph where the locality is low (e.g., not easy to just go to another edge)
        fig, ax = plt.subplots(figsize=(4, 4))
        if self.graph_type == 'chordal':
            G = nx.chordal_cycle_graph(self.V)
            # Draw before inserting self-loops
            labels = True if self.V < 10 else False
            color = 'blue' if self.V > 10 else 'white'
            nx.draw(G, ax=ax, with_labels=labels,
                    node_color=color, node_size=10)
            G.add_edges_from([(i, i) for i in range(self.V)])
            edge_index = np.unique([list(i)[:2]
                                   for i in list(G.edges)], axis=0).tolist()
            m = 0
            while m < len(edge_index):
                edge = edge_index[m]
                k, j = edge
                if [j, k] not in edge_index:
                    edge_index.append([j, k])
                m += 1
        if self.graph_type == 'ER_graph':
            G = nx.erdos_renyi_graph(self.V, 0.02, seed=123, directed=False)
            G.add_edges_from([(i, i) for i in range(self.V)])
            edge_index = []
            for node in range(self.V):
                edge_index += list(G.edges(node))
        self.fig = fig
        edge_index = torch.tensor(edge_index).T.to(device)
        self.edge_index = edge_index
        print(f'{edge_index.shape[1]} number of edges')

    def get_graph_and_Sigma_for_local(self):
        # The KNN graph suggested by Prof. Cheng
        self.knn = False
        if self.knn:
            n, knn = self.V, 2
            np.random.seed(1103)
            T = np.sort(np.random.rand(n))
            X = np.array([[np.cos(np.pi * t), np.sin(np.pi * t)] for t in T])
            X = X + np.random.rand(X.shape[0], X.shape[1]) * 0.05
            A = kneighbors_graph(X, knn, mode='connectivity',
                                 include_self=True).toarray()
            A = A + A.T
            A[A > 0] = 1
            S = np.diag(np.ones(n)) * knn * 2 + A
            D = np.diag(1 / np.sqrt(np.sum(S, axis=1)))
            S = D @ S @ D
            S = (S + S.T) / 2
            # Convert to correlation matrix
            diag = np.sqrt(np.diag(np.diag(S)))
            gaid = np.linalg.inv(diag)
            S = gaid @ S @ gaid
            # NOTE, keep this small, as o/w S_inv would not be local
            offset = np.min(S[S > 0]) / n
            S[0, 1] += offset
            S[1, 0] += offset
            S[n - 2, n - 1] -= offset
            S[n - 1, n - 2] -= offset
            S_inv = np.linalg.inv(S)
            rows, cols = np.where(A == 1)
            edges = list(zip(rows.tolist(), cols.tolist()))
            self.edge_index = torch.tensor(
                [list(i) for i in edges]).T.to(device)
        else:
            self.edge_index = torch.tensor(
                [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]).to(device)
            rho, rho1 = 0.6, -0.4
            S = np.array([[1, rho, 0], [rho, 1, rho1], [0, rho1, 1]])
            S_inv = np.linalg.inv(S)
            print(S_inv)  # Check if "local"
            X = np.zeros((3, 2))
        fig, ax = plt.subplots(2, 2, figsize=(
            8, 8), constrained_layout=True)
        gr = nx.Graph()
        gr.add_edges_from(self.edge_index.T.tolist())
        nx.draw(gr, ax=ax[0, 0], with_labels=True, node_color='white')
        ax[0, 1].plot(X[:, 0], X[:, 1], 'o')
        ax[1, 0].matshow(S)
        ax[1, 0].set_title(r'Corr of $\Sigma$')
        c = ax[1, 1].matshow(S_inv)
        ax[1, 1].set_title(r'Corr of $\Sigma^{-1}$')
        cbar_ax = fig.add_axes([1, 0.025, 0.1, 0.4])
        plt.colorbar(c, cax=cbar_ax)
        plt.show()
        plt.close()
        fig, ax = plt.subplots(figsize=(4, 4))
        gr = nx.Graph()
        gr.add_edges_from(self.edge_index.T.tolist())
        nx.draw(gr, ax=ax, with_labels=True, node_color='white')
        self.fig = fig
        self.Sigma = torch.from_numpy(S).float().to(device)


def isPrime(n):
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


'''2. Real data helpers, Traffic '''


class trffic_data():
    def __init__(self, d):
        '''
        Input:
            d here means how long in the past we look at each node. It is thus the in-channel dimension
        '''
        self.d = d

    def get_traffic_train_test(self, num_neighbor=3, sub=False):
        '''
            Description:
                Data are available hourly, with Yt,i = 1 (resp. 2) if the current traffic flow lies outside the upper (resp. lower) 90% quantile over the past four days of traffic flow of its nearest four neighbors based on sensor proximity.
        '''
        d = self.d
        # Traffic flow multi-class detection
        with open(f'flow_frame_train_0.7_no_drop_data.p', 'rb') as fp:
            Xtrain = pickle.load(fp)
        with open(f'flow_frame_test_0.7_no_drop_data.p', 'rb') as fp:
            Xtest = pickle.load(fp).to_numpy()
        with open(f'true_anomalies.p', 'rb') as fp:
            Yvals = pickle.load(fp).to_numpy()
        # Define edge index
        sensors = np.array(list(Xtrain.columns))
        Xtrain = Xtrain.to_numpy()
        scaler = StandardScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.fit_transform(Xtest)
        Ytrain = Yvals[:Xtrain.shape[0], :]
        Ytest = Yvals[Xtrain.shape[0]:, :]
        if sub:
            N = int(Xtrain.shape[0] / 2)  # 50% or /2 already pretty good
            N1 = int(Xtest.shape[0] / 2)
            Xtrain = Xtrain[-N:]
            Xtest = Xtest[:N1]
            Ytrain = Ytrain[-N:]
            Ytest = Ytest[:N1]
        with open(f'sensor_neighbors.p', 'rb') as fp:
            neighbor_dict = pickle.load(fp)
        # # Randomly select 15 nodes
        # np.random.seed(1103)
        # chosen_nodes = np.random.choice(len(sensors), 15, replace=False)
        # sensors = sensors[chosen_nodes]
        # Xtrain, Xtest, Ytrain, Ytest = Xtrain[:, chosen_nodes], Xtest[:,
        #                                                               chosen_nodes], Ytrain[:, chosen_nodes], Ytest[:, chosen_nodes]
        sensors_dict = {i: j for (i, j) in zip(sensors, range(len(sensors)))}
        edge_index = []
        # num_neighbor = 3
        for k, sensor in enumerate(sensors):
            neighbors = neighbor_dict[sensor]
            num_n = 0
            for p in range(len(sensors)):
                if num_n >= num_neighbor:
                    break
                if neighbors[p] in sensors_dict.keys():
                    edge_index.append([k, sensors_dict[neighbors[p]]])
                    num_n += 1
        edge_index = torch.from_numpy(np.array(edge_index).T).type(torch.long)
        # Define graphs, similarly as the solar data
        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
        for t in range(d - 1, Xtrain.shape[0]):
            X_train.append(
                np.flip(Xtrain[t - d + 1:t + 1].T, 1))
            Y_train.append(Ytrain[t])
        for t in range(Xtest.shape[0]):
            if t < d - 1:
                temp = np.c_[np.flip(Xtest[:t + 1].T, 1),
                             np.flip(Xtrain[-(d - t) + 1:].T, 1)]
            else:
                temp = np.flip(Xtest[t - d + 1:t + 1].T, 1)
            X_test.append(temp)
            Y_test.append(Ytest[t])
        X_train = torch.stack([torch.from_numpy(val.copy()).type(
            torch.FloatTensor) for val in X_train])
        X_test = torch.stack([torch.from_numpy(val.copy()).type(
            torch.FloatTensor) for val in X_test])
        Y_train = torch.stack([torch.from_numpy(val.copy()).type(
            torch.FloatTensor) for val in Y_train])
        Y_test = torch.stack([torch.from_numpy(val.copy()).type(
            torch.FloatTensor) for val in Y_test])
        Y_train[Y_train == 2] = 0
        Y_test[Y_test == 2] = 0
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.edge_index = X_train, X_test, Y_train, Y_test, edge_index

    def plot_traffic(self):
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['figure.titlesize'] = 28
        fig, axs = plt.subplots(3, 1, figsize=(
            3, 7), constrained_layout=True)
        G = nx.Graph()
        G.add_edges_from(self.edge_index.cpu().detach().numpy().T)
        pos = nx.circular_layout(G)
        i = 0
        nx.draw(G, pos, ax=axs[i], with_labels=True, node_color='lightblue')
        N, N1 = self.Y_train_sub.numel(), self.Y_test_sub.numel()
        colors = np.repeat('black', N)
        colors1 = np.repeat('black', N1)
        colors[(self.Y_train_sub.flatten() == 1).cpu(
        ).detach().numpy().flatten()] = 'red'
        colors1[(self.Y_test_sub.flatten() == 1).cpu(
        ).detach().numpy().flatten()] = 'red'
        Xtrain, Xtest = self.X_train_sub.flatten(
            start_dim=0, end_dim=1), self.X_test_sub.flatten(start_dim=0, end_dim=1)
        i += 1
        axs[i].scatter(Xtrain[:, 0], Xtrain[:, 1], s=1, color=colors)
        # axs[0].plot(Xtrain[:, 0], Xtrain[:, 1],
        #             linestyle='dashed', linewidth=0.075)
        axs[i].set_title('Train X')
        i += 1
        axs[i].scatter(Xtest[:, 0], Xtest[:, 1], s=1, color=colors1)
        # axs[1].plot(Xtest[:, 0], Xtest[:, 1],
        #             linestyle='dashed', linewidth=0.075)
        axs[i].set_title('Test X')
        self.fig = fig
        plt.show()
        plt.close()

    def select_Y(self, Y_rows, train=True):
        if train:
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        if Y_rows is None:
            if train:
                self.X_train_sub, self.Y_train_sub = X, Y
            else:
                self.X_test_sub, self.Y_test_sub = X, Y
        else:
            for i, Y_row in enumerate(Y_rows):
                idx_temp = (Y == Y_row).all(dim=1).to(device)
                if i == 0:
                    idx = idx_temp.clone()
                else:
                    idx = torch.logical_or(idx, idx_temp)
            if train:
                self.X_train_sub, self.Y_train_sub = X[idx], Y[idx]
            else:
                self.X_test_sub, self.Y_test_sub = X[idx], Y[idx]


''' 2. Real-data helpers, Solar '''


class solar_data():
    def __init__(self, num_obs_per_day, city):
        self.num_obs_per_day = num_obs_per_day
        self.city = city
        self.V = 10 if self.city == 'CA' else 0
        self.C = 2

    def get_solar(self):
        graph_connect = {'CA': False, 'LA': True}
        DHI_2017 = get_DHI(self.city, '2017', self.V, self.num_obs_per_day)
        DHI_2018 = get_DHI(self.city, '2018', self.V, self.num_obs_per_day)
        DHI_full = np.r_[DHI_2017, DHI_2018]
        T = DHI_full.shape[0]
        N = int(T * 3 / 4)
        DHI_train, DHI_test = DHI_full[:N], DHI_full[N:]
        # Anomalies have same frequency as data
        get_anomaly(DHI_full, self.city, N)
        train_anom = np.loadtxt(
            f'{self.city}_anomalies_train.csv', delimiter=',')
        test_anom = np.loadtxt(
            f'{self.city}_anomalies_test.csv', delimiter=',')
        X_train, X_test, Y_train, Y_test = get_solar_train_test(
            DHI_train, DHI_test, train_anom, test_anom, d=self.C)
        # Each has dimension (N-by-V-by-C), where the response is only N-by-V
        X_train = torch.stack([torch.from_numpy(val.copy()).type(
            torch.FloatTensor) for val in X_train])
        X_test = torch.stack([torch.from_numpy(val.copy()).type(
            torch.FloatTensor) for val in X_test])
        Y_train = torch.stack([torch.from_numpy(val.copy()).type(
            torch.FloatTensor) for val in Y_train])
        Y_test = torch.stack([torch.from_numpy(val.copy()).type(
            torch.FloatTensor) for val in Y_test])
        count_train = np.unique(Y_train.numpy(), return_counts=True)[1]
        count_test = np.unique(Y_test.numpy(), return_counts=True)[1]
        print(
            f'#1/#0 in training data is {count_train[1]/count_train[0]}')
        print(f'#1/#0 in test data is {count_test[1]/count_test[0]}')
        fully_connected = graph_connect[self.city]
        edge_index = get_edge_list(
            Y_train, self.V, fully_connected)
        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test
        self.edge_index = edge_index

    def plot_solar(self):
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['figure.titlesize'] = 28
        fig, axs = plt.subplots(3, 1, figsize=(
            3, 7), constrained_layout=True)
        G = nx.Graph()
        G.add_edges_from(self.edge_index.cpu().detach().numpy().T)
        pos = nx.circular_layout(G)
        i = 0
        nx.draw(G, pos, ax=axs[i], with_labels=True, node_color='lightblue')
        N, N1 = self.Y_train.numel(), self.Y_test.numel()
        colors = np.repeat('black', N)
        colors1 = np.repeat('black', N1)
        colors[(self.Y_train.flatten() == 1).cpu(
        ).detach().numpy().flatten()] = 'red'
        colors1[(self.Y_test.flatten() == 1).cpu(
        ).detach().numpy().flatten()] = 'red'
        Xtrain, Xtest = self.X_train.flatten(
            start_dim=0, end_dim=1), self.X_test.flatten(start_dim=0, end_dim=1)
        i += 1
        axs[i].scatter(Xtrain[:, 0], Xtrain[:, 1], s=1, color=colors)
        # axs[0].plot(Xtrain[:, 0], Xtrain[:, 1],
        #             linestyle='dashed', linewidth=0.075)
        axs[i].set_title('Train X')
        i += 1
        axs[i].scatter(Xtest[:, 0], Xtest[:, 1], s=1, color=colors1)
        # axs[1].plot(Xtest[:, 0], Xtest[:, 1],
        #             linestyle='dashed', linewidth=0.075)
        axs[i].set_title('Test X')
        self.fig = fig
        plt.show()
        plt.close()


def get_anomaly(raw_data, city, N):
    # NOTE: just run once is enough.
    T, V = raw_data.shape
    anomalies = np.zeros((T, V))
    # window = 15  # Used to be 30
    for l in range(V):
        for t in range(1, T):
            # past_window = np.arange(max(0, t-2*window), t, 2, dtype=int)
            # # past_at_l = raw_data[past_window, l]
            # past_at_l = raw_data[past_window]
            # Q1, Q3 = np.percentile(past_at_l, 25), np.percentile(past_at_l, 75)
            # IQR = Q3-Q1
            # lower_end1, lower_end2 = np.percentile(
            #     past_at_l, 5), np.percentile(past_at_l, 10)
            rate_inc = (raw_data[t, l] - raw_data[t - 1, l]
                        ) / raw_data[t - 1, l]
            # if raw_data[t, l] < Q1-IQR:
            # NOTE: use this a little arbitrary rule, as o/w too hard to do.
            if rate_inc > 1 or rate_inc < -0.5 or ((raw_data[t, l] < 40) and (raw_data[t - 1, l] < 35)):
                anomalies[t, l] = 1
    np.savetxt(f'{city}_anomalies_train.csv', anomalies[:N], delimiter=',')
    np.savetxt(f'{city}_anomalies_test.csv', anomalies[N:], delimiter=',')


def get_DHI(city, year, V, num_obs_per_day=2):
    # average_num_obs: how many observations we average over. Default is 24 so it is 12H
    full_data = pd.read_csv(f'{city}_{year}.csv')
    full_data = full_data['DHI'].to_numpy()
    days = 365
    mult = days * 48
    freq = int(24 / (num_obs_per_day - 1))
    T = days * num_obs_per_day
    X_array = np.zeros((T, V))
    for loc in range(V):
        loc_data = full_data[loc * mult:(loc + 1) * mult]
        for d in range(T):
            X_array[d, loc] = np.mean(loc_data[d * freq:d * freq + 24])
    return X_array


def get_solar_train_test(train_DHI, test_DHI, train_anom, test_anom, d=2):
    # d is the dimension of the input signal, which intuitively is the memory depth
    # The training data starts at index d, where each row is X=\omega^-d_t=[\omega_t-1,...,\omega_t-d] \in R^{K-by-d}
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    N, N1 = train_DHI.shape[0], test_DHI.shape[0]
    scaler = StandardScaler()
    train_DHI = scaler.fit_transform(train_DHI)
    test_DHI = scaler.fit_transform(test_DHI)
    for t in range(d - 1, N):
        X_train.append(np.flip(train_DHI[t - d + 1:t + 1], 0).T)
        Y_train.append(train_anom[t])
    for t in range(N1):
        # Use raw DHI, including today
        if t < d - 1:
            temp = temp = np.r_[
                np.flip(test_DHI[:t + 1], 0), np.flip(train_DHI[-(d - t) + 1:], 0)]
        else:
            temp = np.flip(test_DHI[t - d + 1:t + 1], 0)
        X_test.append(temp.T)
        Y_test.append(test_anom[t])
    return [X_train, X_test, Y_train, Y_test]


def get_edge_list(Y_train, n=10, fully_connected=True):
    if fully_connected:
        edge_index = torch.from_numpy(
            np.array([[a, b] for a in range(n) for b in range(n)]).T).type(torch.long)
    else:
        # Infer edge connection in a nearest neighbor fashion, by including connection among node k and all nodes whose training labels are the most similar to k (e.g., in terms of equality). The reason is that this likely indicates influence.
        # Always include itself
        Y_temp = np.array(Y_train)
        edge_index = []
        num_include = 3  # four nodes, including itself
        for k in range(n):
            same_num = np.array([np.sum(Y_temp[:, k] == Y_temp[:, j])
                                 for j in range(Y_temp.shape[1])])
            include_ones = same_num.argsort()[-num_include:][::-1]
            for j in include_ones:
                edge_index.append([k, j])
        # Also, to ensure the connection is symmetric, I included all edges where there was a directed edge before
        print(f'{len(edge_index)} directed edges initially')
        m = 0
        while m < len(edge_index):
            edge = edge_index[m]
            k, j = edge
            if [j, k] in edge_index:
                # print(f'{[j, k]}' in edge')
                m += 1
                continue
            else:
                # print(f'{[j, k]} added b/c {[k, j]}' in graph')
                edge_index.append([j, k])
                m += 1
        print(f'{len(edge_index)} undirected edges after insertion')
        edge_index = torch.from_numpy(np.array(edge_index).T)
    return edge_index


'''3. Simulation non-graph helper and other helpers'''


def gen_image_data(img_mask, N):
    ''' From FFJORD '''
    def sample_data(train_data_size):
        inds = np.random.choice(
            int(probs.shape[0]), int(train_data_size), p=probs)
        m = means[inds]
        samples = np.random.randn(*m.shape) * std + m
        return samples
    img = img_mask
    h, w = img.shape
    xx = np.linspace(-4, 4, w)
    yy = np.linspace(-4, 4, h)
    xx, yy = np.meshgrid(xx, yy)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    means = np.concatenate([xx, yy], 1)
    img = img.max() - img
    probs = img.reshape(-1) / img.sum()
    std = np.array([8 / w / 2, 8 / h / 2])
    X_np = sample_data(N)
    y_np = np.zeros((N, 1))
    X = torch.from_numpy(X_np).float().to(device)
    y = torch.from_numpy(y_np).float().to(device)
    return [X_np, y_np, X, y]


def gen_two_moon_data(N, data_name, noise=0.05, x_offset=0, y_offset=0):
    if data_name == 'two_moon':
        X_np, y_np = make_moons(noise=noise, n_samples=N)
        X_np = StandardScaler().fit_transform(X_np)
        loc0 = y_np == 0  # Top half circle
        loc1 = y_np == 1
        change = np.array([x_offset, -y_offset])
        X_np[loc0] += change
        X_np[loc1] -= change
        X = torch.from_numpy(X_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device)
    elif data_name == 'two_circles':
        X_np, y_np = make_circles(n_samples=N, noise=0.01, factor=0.4)
        X = torch.from_numpy(X_np).float().to(device)
        y = torch.from_numpy(y_np).float().to(device)
    else:
        raise ValueError('Not considered yet')
    return [X_np, y_np, X, y]


def make_a_moon(N, center, radius, sigma, pi0, pi1, class_Y):
    y_np = np.zeros(N)+class_Y
    X_np = np.zeros((N, 2))
    pi0 = np.pi*(pi0/180)
    pi1 = np.pi*(pi1/180)
    q = np.random.uniform(pi0, pi1, size=N)
    X_np[:, 0] = radius*np.cos(q)
    X_np[:, 1] = radius*np.sin(q)
    X_np += center
    noise = np.random.normal(0, sigma, size=X_np.shape)
    X_np += noise
    return X_np, y_np


def make_many_moons(N_moon0, N_moon1, N_moon2, use_larger_r=True):
    sigma = 0.05
    radius = 1
    X_np0, y_np0 = make_a_moon(N_moon0, center=(
        radius, radius/2), radius=1, sigma=sigma, pi0=180, pi1=360, class_Y=0)
    X_np1, y_np1 = make_a_moon(N_moon1, center=(
        0, 0), radius=1, sigma=sigma, pi0=0, pi1=180, class_Y=1)
    large_r = 2*radius
    if use_larger_r:
        large_r = 3*radius
    large_r = 2*radius
    X_np2, y_np2 = make_a_moon(N_moon2, center=(
        radius/2, radius/4), radius=large_r, sigma=sigma, pi0=90, pi1=360, class_Y=2)
    idx = torch.randperm(N_moon0+N_moon1+N_moon2)  # Randomly shuffle
    X_np = np.r_[X_np0, X_np1, X_np2][idx]
    y_np = np.r_[y_np0, y_np1, y_np2][idx]
    X = torch.from_numpy(X_np).float().to(device)
    y = torch.from_numpy(y_np).float().to(device)
    return [X_np, y_np, X, y]

# def make_many_moons(N_per_moon, number_of_moons=3, sigma=0.03, radius=0.75, y_shift=-2):
#     moons = []
#     for y in range(number_of_moons):
#         q = np.random.uniform(0, np.pi, size=N_per_moon)
#         if y % 2 == 0:
#             factor = 1
#         else:
#             factor = -1
#         moon = np.zeros((N_per_moon, 3))
#         moon[:, 0] = (radius * np.cos(q)) + y
#         moon[:, 1] = (radius * np.sin(q) * factor) + (factor == -1) * y_shift
#         moon[:, 2] = y
#         moons.append(moon)
#         noise = np.random.normal(0, sigma, size=moon[:, :2].shape)
#         moon[:, :2] += noise
#     moons = np.concatenate(moons)
#     idx = torch.randperm(moons.shape[0])  # Randomly shuffle
#     moons = moons[idx]
#     X_np, y_np = moons[:, :2], moons[:, 2]
#     X = torch.from_numpy(X_np).float().to(device)
#     y = torch.from_numpy(y_np).float().to(device)
#     return [X_np, y_np, X, y]


def draw_graph(edge_index, edge_index_est, graph_type, overleaf_path):
    G = nx.Graph()
    G.add_edges_from(edge_index.cpu().detach().numpy().T)
    pos = nx.circular_layout(G)
    fig_network, ax1 = plt.subplots(1, 2, figsize=(8, 4))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax1[0])
    # ax1[0].set_title('True Graph')
    G = nx.Graph()
    G.add_edges_from(edge_index_est.cpu().detach().numpy().T)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax1[1])
    # ax1[1].set_title('Estimated Graph')
    fig_network.savefig(f'{overleaf_path}simulated_graph_{graph_type}.pdf',
                        dpi=200, bbox_inches='tight', pad_inches=0)


verts = [
    (-2.4142, 1.),
    (-1., 2.4142),
    (1.,  2.4142),
    (2.4142,  1.),
    (2.4142, -1.),
    (1., -2.4142),
    (-1., -2.4142),
    (-2.4142, -1.)
]
label_maps = {
    'all':  [0, 1, 2, 3, 4, 5, 6, 7],
    'some': [0, 0, 1, 1, 2, 2, 3, 3],
    'none': [0, 0, 0, 0, 0, 0, 0, 0],
}


def gen_8_gaussian_data(labels, N, random_state=0):
    # N denotes number of obs for all 8 Gaussian
    np.random.seed(random_state)
    mapping = label_maps[labels]

    pos = np.random.normal(size=(N, 2), scale=0.2)
    labels = np.zeros((N, 8))
    n = N//8

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.

    shuffling = np.random.permutation(N)
    pos = torch.tensor(pos[shuffling], dtype=torch.float)
    labels = torch.tensor(labels[shuffling], dtype=torch.float)

    return pos, labels


##########


##########
##########
##########
##########
