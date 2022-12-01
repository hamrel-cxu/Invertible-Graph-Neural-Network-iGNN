import numpy as np
import torch
import os
import time as time
import pdb
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
import visualize_new_code as viz
import os
import shutil
import sys
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import humanize
import psutil
import GPUtil
import pdb
import pandas as pd
import utils_data_new_code as utils_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cvt(x): return x.to(device)


class IResNet_training_on_graph():
    def __init__(self, models, mod_args, train_args, data_args, X_train, Y_train, X_test=None, Y_test=None):
        self.model, self.gen_net = models
        self.dim, self.nblocks = mod_args
        self.optimizer, self.optimizer_gen_net, self.optim_classify, self.mu, self.gamma, self.epochs, self.scheduler, self.resume_checkpoint = train_args
        self.edge_index, self.edge_weight, self.batch_size, self.data_name, self.num_viz, self.num_to_plot = data_args
        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test
        self.N, self.V, self.C = self.X_train.shape
        # NOTE, these parameters are only updated when we access trained models afterwards for different visualizations
        # self.viz determines if transport cost is visualized
        self.cpu_load, self.viz = False, False
        # The criterion at which we stop training (default is <=0.01% error in consecutive training generative loss)
        self.stop_criterion = 1e-7
        # This is true ONLY when we visualize at random times our trained model
        self.final_viz = False
        self.plot_sub = False
        self.prefix = ''  # For file saving
        self.CINN_obj = ''  # For CINN training
        # Record transport cost over blocks at the end of .gif trajectory
        self.transport_cost_XtoH_ls, self.transport_cost_HtoX_ls = [], []
        # Sometimes we try different Y by fixing them in training, in which case it is the best to not save the reindex
        self.save_reindex = True if self.num_viz > 1 else False
        self.non_invertible_ls = []
        self.X_dist = None
        self.err_on_dist = 0
        self.err_off_dist = 0
        # If higher, repulsion loss more. Default 0 means not using, where grad of mu(H|Y) model is also disabled
        self.load_losses = True
        self.r_mult = 0
        self.two_moon_noise, self.x_offset, self.y_offset = 0.05, 0, 0
        self.same_row = False
    '''Train IResNet (our method)'''

    def all_together(self):
        self.load_from_checkpoint()  # Load previous models from file
        self.get_H_cond_Y()  # Get H|Y
        while self.epoch < self.epochs:
            if device.type == 'cuda':
                # Useful to avoid GPU allocation excess
                torch.cuda.empty_cache()
            if self.X_dist is not None and self.epoch > 0:
                # regenerate samples
                self.regenerate_sample()
            # Check inversion
            print(f"LR is {self.optimizer.param_groups[0]['lr']}")
            self.check_model_inversion()
            # Visualize generation
            self.viz_generation()
            if self.Y_test is not None:
                self.viz_generation(viz_train=False)
            start_epoch = time.time()
            print(f'Epoch {self.epoch}')
            loss_g_ave, loss_w2_ave, loss_c_ave, classify_error_ave = self.batch_training(
                train=True)
            mem_report()  # Print GPU usage and availability
            self.loss_g_ls_train.append(loss_g_ave)
            self.loss_w2_ls_train.append(loss_w2_ave)
            self.loss_c_ls_train.append(loss_c_ave)
            if len(self.loss_g_ls_train) > 10 and np.abs((self.loss_g_ls_train[-1]-self.loss_g_ls_train[-2])/self.loss_g_ls_train[-2]) < self.stop_criterion:
                # If consecutive dec. less than X%, then just break.
                break
            self.classify_error_ls_train.append(classify_error_ave)
            if self.Y_test is not None:
                with torch.no_grad():
                    # NOTE, this is important, as o/w somehow memory accumulates and GPU depletes quickly
                    loss_g_ave, loss_w2_ave, loss_c_ave, classify_error_ave = self.batch_training(
                        train=False)
            else:
                loss_g_ave, loss_w2_ave, loss_c_ave, classify_error_ave = 0, 0, 0, 1
            self.loss_g_ls_test.append(loss_g_ave)
            self.loss_w2_ls_test.append(loss_w2_ave)
            self.loss_c_ls_test.append(loss_c_ave)
            self.classify_error_ls_test.append(classify_error_ave)
            print(
                f'After Epoch {self.epoch}: \n Training loss_g is {self.loss_g_ls_train[-1]} & Test loss_g is {self.loss_g_ls_test[-1]} \n Training loss_c is {self.loss_c_ls_train[-1]} & Test loss_c is {self.loss_c_ls_test[-1]} \n Training classify error is {self.classify_error_ls_train[-1]} & Test classify error is {self.classify_error_ls_test[-1]}')
            print(f'Epoch {self.epoch} takes {time.time()-start_epoch} secs.')
            self.viz_losses()
            self.save_checkpoint()
            self.epoch += 1

    def get_repulsion_loss(self):

        def log_barrier(x, c=2):
            '''
                log-barrier function at delta
                Note, this has been automatically adjusted so that as long as initial H|Y are pairwise disjoint
                The "delta" term would take into account this
                Furthermore, if x
            '''
            const = torch.ones(1).to(device)
            diff = x-c
            delta = 1e-2
            diff[diff > delta] = torch.ones(1).to(device)
            return -torch.log(diff)
        gnet = self.gen_net
        K = gnet.fc.in_features
        delta = 2*norm().ppf(0.9999)*self.sigma_HY
        weights = gnet(torch.eye(K).to(device))
        pwise_dist = torch.cdist(weights, weights, p=2)
        idx = torch.triu_indices(*pwise_dist.shape, offset=1)
        distances = pwise_dist[idx[0], idx[1]]
        repulsion_loss = log_barrier(distances, c=delta).sum()
        return self.r_mult*repulsion_loss

    def batch_training(self, train=True):
        if train:
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        loss_g_tot = 0
        loss_w2_tot = 0
        loss_c_tot = 0
        numcorrect_tot = 0
        N_tmp = len(Y)
        batch_idxs = np.arange(N_tmp)
        for batch in range(int(np.ceil(N_tmp / self.batch_size))):
            batch_idx = batch_idxs[batch
                                   * self.batch_size:np.min([(batch + 1) * self.batch_size, N_tmp])]
            loss_g, loss_w2, loss_g_likelihood = self.get_L_g(batch_idx, X, Y)
            mu_repulsion_loss = self.get_repulsion_loss()
            if batch == 0:
                mem_report()  # Print GPU usage and availability
            loss_g_tot += loss_g_likelihood
            loss_w2_tot += loss_w2
            if self.optim_classify:
                # Feed in a tensor of shape N-by-nC
                loss_c, numcorrect = self.get_L_c(batch_idx, X, Y)

            else:
                loss_c = torch.zeros(1).to(device)
                numcorrect = torch.zeros(1).to(device)
            numcorrect_tot += numcorrect
            loss_c_tot += loss_c
            if train:
                self.optimizer.zero_grad()
                self.optimizer_gen_net.zero_grad()
                loss_g_pls_c = loss_g + loss_c + mu_repulsion_loss
                loss_g_pls_c.backward()
                self.optimizer_gen_net.step()
                self.optimizer.step()
                self.get_H_cond_Y()  # This is necessary especially when gen_net is trained, since somethings seems to be freed during training
                print(
                    f'After Epoch {self.epoch} & Batch {batch}: \n Training loss_g is {loss_g} \n Training loss_c is {loss_c}.')
        if self.scheduler is not None:
            self.scheduler.step()
        # if 'two_moon' in self.path:
        #     decay_freq = 5
        #     if self.epoch > 25 and self.epoch % decay_freq == 0:
        #         self.optimizer.param_groups[0]['lr'] /= 1.05
        # if 'two_circles' in self.path:
        #     decay_freq = 5
        #     if self.epoch > 25 and self.epoch % decay_freq == 0:
        #         self.optimizer.param_groups[0]['lr'] /= 1.05
        if 'three_moon' in self.path:
            decay_freq = 5
            if self.epoch > 25 and self.epoch % decay_freq == 0:
                self.optimizer.param_groups[0]['lr'] /= 1.025
        if '3node' in self.path:
            decay_freq = 10
            if self.epoch > 50 and self.epoch % decay_freq == 0:
                self.optimizer.param_groups[0]['lr'] /= 1.01
        if 'CA_solar' in self.path:
            decay_freq = 10
            if self.epoch > 50 and self.epoch % decay_freq == 0:
                self.optimizer.param_groups[0]['lr'] /= 1.01
        if 'traffic' in self.path:
            decay_freq = 10
            if self.epoch > 50 and self.epoch % decay_freq == 0:
                self.optimizer.param_groups[0]['lr'] /= 1.01
            else:
                print(
                    f'After Epoch {self.epoch} & Batch {batch}: \n Test loss_g is {loss_g} \n Test loss_c is {loss_c}.')
        return [loss_g_tot.item()/(batch+1), loss_w2_tot.item()/(batch+1), loss_c_tot.item()/(batch+1), 1 - numcorrect_tot.item() / Y.numel()]

    def get_L_g(self, batch_idx, X, Y):
        # TODO: simplify this :)
        # Return the L_g on X_train[batch_idx] by using the change-of-variable formula
        X_batch = X[batch_idx].flatten(start_dim=1)
        H_pred, log_det, transport_cost = self.model(
            X_batch, self.edge_index, self.edge_weight)
        # Reshape the tensor to N-by-n-by-C
        batch_size = len(batch_idx)
        H_pred = H_pred.reshape(batch_size, self.V, self.C)
        C_tmp = self.C
        if self.C > 2:
            # We need to evaluate likelihood in R^2, so reshape stuff.
            V_tmp = int(self.C/2)
            C_tmp = 2
            H_pred = H_pred.reshape(batch_size, V_tmp, C_tmp)
        tot_element = Y[batch_idx].numel()
        # We take sum so that the log-likelihood is average over graph, not each node.
        if '8_gaussian' in self.path:
            num_1 = (Y[batch_idx] == 1).sum().item()
            num_2 = (Y[batch_idx] == 2).sum().item()
            num_3 = (Y[batch_idx] == 3).sum().item()
        elif 'three_moon' in self.path:
            num_1 = (Y[batch_idx] == 1).sum().item()
            num_2 = (Y[batch_idx] == 2).sum().item()
            num_3 = 0
        else:
            num_1 = (Y[batch_idx] == 1).sum().item()
            num_2, num_3 = 0, 0
        # Evaluate log like by class, where we assume at most 4 classes of Y exist
        if num_1 > 0:
            one_idx = (Y[batch_idx] == 1).unsqueeze(-1).repeat(1, 1, C_tmp)
            one_dim = int(one_idx.sum().item() / C_tmp)
            log_pH1 = self.base_dist1.log_prob(
                H_pred[one_idx].reshape(one_dim, C_tmp)).sum()
        else:
            log_pH1 = 0
        if num_2 > 0:
            two_idx = (Y[batch_idx] == 2).unsqueeze(-1).repeat(1, 1, C_tmp)
            two_dim = int(two_idx.sum().item() / C_tmp)
            log_pH2 = self.base_dist2.log_prob(
                H_pred[two_idx].reshape(two_dim, C_tmp)).sum()
        else:
            log_pH2 = 0
        if num_3 > 0:
            three_idx = (Y[batch_idx] == 3).unsqueeze(-1).repeat(1, 1, C_tmp)
            three_dim = int(three_idx.sum().item() / C_tmp)
            log_pH3 = self.base_dist3.log_prob(
                H_pred[three_idx].reshape(three_dim, C_tmp)).sum()
        else:
            log_pH3 = 0
        if tot_element - num_1-num_2-num_3 > 0:
            zero_idx = (Y[batch_idx] == 0).unsqueeze(-1).repeat(1, 1, C_tmp)
            zero_dim = int(zero_idx.sum().item() / C_tmp)
            log_pH0 = self.base_dist0.log_prob(
                H_pred[zero_idx].reshape(zero_dim, C_tmp)).sum()
        else:
            log_pH0 = 0
        log_pH = log_pH0 + log_pH1 + log_pH2 + log_pH3
        if self.C == 1 and ('one_Cheb' in self.path) or ('one_L3' in self.path):
            # Flow with ChebNet directly, so no need W2 regularization
            self.viz = True
            if 'W2' in self.path:
                self.viz = False
        likelihood = (log_pH + log_det)/batch_size
        if self.viz:
            # Need not transport cost
            w2_loss = torch.zeros(1).to(device)
        else:
            # For training, add transport cost
            w2_loss = - self.gamma*transport_cost/batch_size
        logpx = likelihood + w2_loss
        return -logpx, -w2_loss, -likelihood

    def get_L_c(self, batch_idx, X, Y):
        X_batch = X[batch_idx].flatten(start_dim=1)
        H_pred = self.model.forward(
            X_batch, self.edge_index, self.edge_weight, logdet=False)
        batch_size = len(batch_idx)
        if self.C > 2:
            # NOTE: this is because FC treated graph example in R^V-x-C as a vector in \R^V-by-C, so that we need reshaping for visualization
            V = int(self.C/2)
            H_pred = H_pred.reshape(batch_size, V, 2)
        else:
            H_pred = H_pred.reshape(batch_size, self.V, self.C)
        Y_pred = self.model.classification(H_pred)
        Y_pred = Y_pred.flatten(start_dim=0, end_dim=1)
        Y_target = Y[batch_idx].flatten()
        if Y.max() > 1:
            # Multi-class
            loss_f = torch.nn.CrossEntropyLoss()
            Y_target = Y_target.long()
            Y_pred_label = Y_pred.argmax(dim=1)
        else:
            loss_f = torch.nn.BCELoss()
            Y_pred = torch.sigmoid(Y_pred.flatten())
            Y_pred_label = Y_pred.round()
        loss_c = self.mu * loss_f(Y_pred, Y_target)
        numcorrect = (Y_pred_label == Y_target).sum()
        return [loss_c, numcorrect]

    def regenerate_sample(self):
        if self.X_dist in ['two_moon', 'two_circles']:
            # Regenerate simulated data
            N, V, C = self.X_train.shape
            X_np, y_np, X_train, Y_train = utils_data.gen_two_moon_data(
                N, self.X_dist, self.two_moon_noise, self.x_offset, self.y_offset)
            self.X_train, self.Y_train = X_train.reshape(
                N, V, C), Y_train.reshape(N, V)
        elif self.X_dist == 'three_moon':
            # Regenerate simulated data
            N, V, C = self.X_train.shape
            N_per_moon = int(N/3)
            X_np, y_np, X_train, Y_train = utils_data.make_many_moons(
                N_per_moon, N_per_moon, N_per_moon)
            self.X_train, self.Y_train = X_train.reshape(
                N, V, C), Y_train.reshape(N, V)
        elif self.X_dist == 'three_node_graph':
            import itertools
            data_generator = self.data_generator
            data_generator.get_full_data()
            Y_rows = torch.tensor(
                list(itertools.product(*[[0, 1], [0, 1], [0, 1]])))
            data_generator.select_Y(Y_rows)
            self.X_train, self.Y_train = cvt(
                data_generator.X_full), cvt(data_generator.Y_full)
            data_generator.get_full_data()
            data_generator.select_Y(Y_rows)
            self.X_test, self.Y_test = cvt(
                data_generator.X_full), cvt(data_generator.Y_full)
        elif self.X_dist == 'many_node_graph':
            data_generator = self.data_generator
            data_generator.get_X_Y()
            self.X_train, self.Y_train = data_generator.X_full, data_generator.Y_full
        else:
            # Regenerate samples in the graph case for simulation
            self.X_train, self.Y_train = utils_data.quick_sample(
                self.X_dist, self.X_test.shape[0], self.V)
            self.X_test, self.Y_test = utils_data.quick_sample(
                self.X_dist, self.X_test.shape[0], self.V)

    '''Train CGAN or CINN for comparison'''

    def all_competitor_together(self):
        self.load_from_competitors_checkpoint()
        while self.epoch < self.epochs:
            if device.type == 'cuda':
                # Useful to avoid GPU allocation excess
                torch.cuda.empty_cache()
            # Visualize generation
            self.viz_competitors_generation()
            if self.Y_test is not None:
                self.viz_competitors_generation(viz_train=False)
            if 'CGAN' in self.path:
                if len(self.loss_GAN_train) > 10 and np.abs((self.loss_GAN_train[-1]-self.loss_GAN_train[-2])/self.loss_GAN_train[-2]) < self.stop_criterion:
                    # If consecutive dec. less than X%, then just break.
                    break
                loss_GAN_tot = self.batch_CGAN_training(train=True)
                self.loss_GAN_train.append(loss_GAN_tot)
                mem_report()  # Print GPU usage and availability
                if self.Y_test is not None:
                    with torch.no_grad():
                        loss_GAN_tot = self.batch_CGAN_training(train=False)
                        self.loss_GAN_test.append(loss_GAN_tot)
                else:
                    self.loss_GAN_test.append(0)
                print(
                    f'After Epoch {self.epoch}: CGAN train loss is {self.loss_GAN_train[-1]} \n CGAN test loss is {self.loss_GAN_test[-1]}')
            if 'CINN' in self.path:
                if len(self.loss_CINN_train) > 10 and np.abs((self.loss_CINN_train[-1]-self.loss_CINN_train[-2])/self.loss_CINN_train[-2]) < self.stop_criterion:
                    # If consecutive dec. less than X%, then just break.
                    break
                loss_CINN_tot = self.batch_CINN_training(train=True)
                self.loss_CINN_train.append(loss_CINN_tot)
                mem_report()  # Print GPU usage and availability
                if self.Y_test is not None:
                    with torch.no_grad():
                        loss_CINN_tot = self.batch_CINN_training(train=False)
                        self.loss_CINN_test.append(loss_CINN_tot)
                else:
                    self.loss_CINN_test.append(0)
                print(
                    f'After Epoch {self.epoch}: CINN train loss is {self.loss_CINN_train[-1]} \n CINN test loss is {self.loss_CINN_test[-1]}')
            self.viz_losses_competitors()
            self.save_competitors_checkpoint()
            self.epoch += 1

    def batch_CGAN_training(self, train):
        if train:
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        loss_GAN_tot = 0
        V, C = self.V, self.C
        N_tmp = len(Y)
        batch_idxs = np.arange(N_tmp)
        for batch in range(int(np.ceil(N_tmp / self.batch_size))):
            # print(batch_idxs[-1])
            batch_idx = batch_idxs[batch
                                   * self.batch_size:np.min([(batch + 1) * self.batch_size, N_tmp])]
            num_b = len(batch_idx)
            X_batch = X[batch_idx]
            Y_batch = Y[batch_idx]
            nclasses = 4 if '8_gaussian' in self.path else 2
            Y_batch = F.one_hot(Y_batch.long(), num_classes=nclasses)
            torch.manual_seed(1103)
            noise_z = torch.randn(num_b, V, C).to(device)
            data_for_G = torch.cat((Y_batch, noise_z), -1).detach()
            X_hat_batch = self.net_G(
                data_for_G, self.edge_index, self.edge_weight)
            data_for_D_real = torch.cat((Y_batch, X_batch), -1).detach()
            data_for_D_fake = torch.cat((Y_batch, X_hat_batch), -1).detach()
            # First Discriminator loss
            self.net_D.set_requires_grad(True)
            pred_true = self.net_D(
                data_for_D_real, self.edge_index, self.edge_weight)
            pred_fake = self.net_D(
                data_for_D_fake, self.edge_index, self.edge_weight)
            # Original GAN loss
            # 0.5 is used to "slow down" rate of D learning relative to G
            # Here, D tries to maximize the likelihood of true data and minimize the likelihood of fake
            loss_D = -0.5*(torch.log(pred_true)
                           + torch.log(1-pred_fake)).mean()
            # # Wasserstain GAN loss
            # # Maximize the difference D(real)-D(fake)
            # loss_D = -0.5*(pred_true - pred_fake).mean()
            if train:
                self.optimizer_D.zero_grad()
                loss_D.backward()
                self.optimizer_D.step()
            # Then Generative loss
            self.net_D.set_requires_grad(False)
            data_for_D_fake = torch.cat((Y_batch, X_hat_batch), -1)
            pred_fake = self.net_D(
                data_for_D_fake, self.edge_index, self.edge_weight)
            # Original GAN loss
            # Here, G tries to maximize the likelihood of D thinking the prediction of G is "real"
            loss_G_1 = -torch.log(pred_fake).mean()
            # # Wasserstain GAN
            # loss_G_1 = -pred_fake.mean()
            lam = 1  # For l2 loss between truth and fake
            loss_G_2 = torch.linalg.norm(
                X_batch-X_hat_batch, ord=2, dim=(1, 2)).mean()
            # loss_G_2 = 0
            loss_G = loss_G_1 + lam*loss_G_2
            if train:
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()
            loss_GAN_tot += (loss_D.item()+loss_G.item())*num_b
        return loss_GAN_tot/N_tmp

    def batch_CINN_training(self, train):
        '''
            Code adopted from https://github.com/VLL-HD/analyzing_inverse_problems/tree/master/toy_8-modes of "Analyzing Inverse Problems with Invertible Neural Networks", ICLR 2019

            Basically, it optimizers MMD in forward y, z direction and backward x direction, AS WELL AS the originally fitted errors in the padded y-z space and x space.
        '''
        if train:
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        loss_CINN_tot = 0
        V, C = self.V, self.C
        N_tmp = len(Y)
        batch_idxs = np.arange(N_tmp)
        for _ in ['hyper-parameters']:
            # For 8 Gaussian
            lambd_predict = 3.
            lambd_latent = 300.
            lambd_rev = 400.
            loss_factor = min(
                1., 2. * 0.002**(1. - (float(self.epoch) / self.epochs)))
            y_noise_scale = 1e-1
            zeros_noise_scale = 5e-2
        ndim_x, ndim_y, ndim_tot = self.ndim_x, self.ndim_y, self.ndim_tot
        ndim_z = ndim_x
        loss_backward = MMD_multiscale
        loss_latent = MMD_multiscale
        loss_fit = fit  # MSE
        C = self.C
        base_dist = MultivariateNormal(
            torch.zeros(C).to(device), torch.diag(torch.ones(C)).to(device))
        # If MMD on x-space is present from the start, the model can get stuck.
        # Instead, ramp it up exponetially.
        loss_factor = min(
            1., 2. * 0.002**(1. - (float(self.epoch) / self.epochs)))
        for batch in range(int(np.ceil(N_tmp / self.batch_size))):
            # print(batch_idxs[-1])
            batch_idx = batch_idxs[batch
                                   * self.batch_size:np.min([(batch + 1) * self.batch_size, N_tmp])]
            batch_size = len(batch_idx)
            x = X[batch_idx].flatten(start_dim=1)
            if self.V == 1:
                y = F.one_hot(Y[batch_idx].flatten().long(),
                              num_classes=4).float()
            else:
                y = F.one_hot(Y[batch_idx].long(),
                              num_classes=2).float().flatten(start_dim=1)
            # Pad X and Y to the high-dimension
            y_clean = y.clone()
            pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot
                                                     - ndim_y - ndim_z, device=device)
            if 'Nflow' not in self.CINN_obj:
                y += y_noise_scale * \
                    torch.randn(batch_size, ndim_y,
                                dtype=torch.float, device=device)
                pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot
                                                        - ndim_x, device=device)
                x = torch.cat([x, pad_x],  dim=1)
                y = torch.cat(
                    (torch.randn(batch_size, ndim_z, device=device), pad_yz, y), dim=1)
            # Forward step:
            loss_forward = 0
            if 'Nflow' in self.CINN_obj:
                output, log_det = self.model(x, y)
                log_prob = base_dist.log_prob(output.reshape(
                    batch_size, self.V, C)).sum()
                loss_forward -= log_prob
                loss_forward -= log_det.sum()
                loss_forward /= batch_size
            else:
                output, log_det = self.model(x)
                # MMD-based
                # Shorten output, and remove gradients wrt y, for latent loss
                y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)
                loss_y = lambd_predict * \
                    loss_fit(output[:, ndim_z:], y[:, ndim_z:])
                loss_forward += loss_y
                output_block_grad = torch.cat((output[:, :ndim_z],
                                               output[:, -ndim_y:].data), dim=1)

                loss_z = lambd_latent * loss_latent(output_block_grad, y_short)
                loss_forward += loss_z
                # Backward step:
                pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot
                                                         - ndim_y - ndim_z, device=device)
                y = y_clean + y_noise_scale * \
                    torch.randn(batch_size, ndim_y, device=device)

                orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale
                                    * torch.randn(batch_size, ndim_z, device=device))
                y_rev = torch.cat((orig_z_perturbed, pad_yz,
                                   y), dim=1)
                y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,
                                        y), dim=1)

                # output_rev = model(y_rev, rev=True)
                # output_rev_rand = model(y_rev_rand, rev=True)
                # Chen: their code errors
                output_rev = self.model(y_rev, rev=True)[0]
                output_rev_rand = self.model(y_rev_rand, rev=True)[0]
                l_rev = 0
                loss_x_1 = (
                    lambd_rev
                    * loss_factor
                    * loss_backward(output_rev_rand[:, :ndim_x],
                                    x[:, :ndim_x])
                )
                l_rev += loss_x_1
                loss_x_2 = lambd_predict * loss_fit(output_rev, x)
                l_rev += loss_x_2
                loss_CINN_tot += l_rev.data.item()*batch_size
            loss_CINN_tot += loss_forward.data.item()*batch_size
            for param in self.model.parameters():
                # Chen, they did not include this earlier
                if param.grad is None:
                    continue
                param.grad.data.clamp_(-15.00, 15.00)
            if train:
                self.optimizer.zero_grad()
                loss_forward.backward()
                if 'Nflow' not in self.CINN_obj:
                    l_rev.backward()
                self.optimizer.step()
        return loss_CINN_tot/N_tmp

    '''Visulization, loading, and saving for competitors '''

    def viz_losses_competitors(self):
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['font.size'] = 18
        plt.rcParams['figure.titlesize'] = 22
        plt.rcParams['legend.fontsize'] = 18
        if 'CGAN' in self.path:
            loss_train, loss_test = self.loss_GAN_train, self.loss_GAN_test
            title = 'CGAN loss'
        if 'CINN' in self.path:
            loss_train, loss_test = self.loss_CINN_train, self.loss_CINN_test
            title = 'CINN loss'
        if np.mod(self.epoch + 1, 5) or self.final_viz:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot(loss_train, label=r'Training', color='black')
            ax.plot(loss_test, label=r'Test', color='blue')
            ax.set_title(title)
            ax.legend()
            save_prefix = self.path+'/' if self.prefix == '' else self.prefix
            fig.savefig(f'{save_prefix}Losses_epoch{self.epoch+1}.png',
                        dpi=150, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()

    def viz_competitors_generation(self, viz_train=True):
        if viz_train:
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        V, C = self.V, self.C
        Unique_Y, counts_Y = torch.unique(Y, return_counts=True, dim=0)
        counts_Y, idx = torch.sort(counts_Y, descending=True)
        Unique_Y = Unique_Y[idx]
        viz_freq = 5 if self.num_viz == 1 else 15
        if self.epoch > 50 and viz_freq == 5:
            viz_freq = 10
        # if 'CGAN' in self.path:
        #     viz_freq = 50
        if self.final_viz:
            self.two_sample_stat = {}
        if np.mod(self.epoch + 1, viz_freq) == 0 or self.final_viz:
            for pp, Y_row in enumerate(Unique_Y[:self.num_viz]):
                if self.V == 1 and self.C == 2:
                    # Two moon plot
                    Y_row = None
                which_rows = (Y == Y_row).all(
                    dim=1) if Y_row is not None else torch.tensor([True]).repeat(X.shape[0])
                num_to_gen = sum(which_rows).item()
                print(
                    f'Checking generation at \n {Y_row} with {num_to_gen} out of {len(Y)} data')
                X_sub, Y_sub = X[which_rows], Y[which_rows]
                if 'CGAN' in self.path:
                    torch.manual_seed(1103)
                    noise_z = torch.randn(num_to_gen, V, C).to(device)
                    nclasses = 4 if '8_gaussian' in self.path else 2
                    Y_sub = F.one_hot(Y_sub.long(), num_classes=nclasses)
                    with torch.no_grad():
                        X_pred = self.net_G(torch.cat((Y_sub, noise_z), -1),
                                            self.edge_index, self.edge_weight).cpu()
                    H, H_pred = None, None
                if 'CINN' in self.path:
                    noise_z = torch.randn(num_to_gen, V*C).to(device)
                    y_noise_scale = 1e-1
                    zeros_noise_scale = 5e-2
                    ndim_x, ndim_y, ndim_tot = self.ndim_x, self.ndim_y, self.ndim_tot
                    ndim_z = ndim_x
                    x_samps = X_sub.flatten(start_dim=1)
                    if self.V == 1:
                        y_samps = F.one_hot(
                            Y_sub.flatten().long(), num_classes=4).float()
                    else:
                        y_samps = F.one_hot(
                            Y_sub.long(), num_classes=2).float().flatten(start_dim=1)
                    z = torch.randn(num_to_gen, ndim_z).to(device)
                    if 'Nflow' not in self.CINN_obj:
                        x_pad = torch.zeros(
                            num_to_gen, ndim_tot - ndim_x).to(device)
                        x_samps = torch.cat((x_samps, x_pad), dim=1).to(device)
                        y_samps += y_noise_scale * \
                            torch.randn(num_to_gen, ndim_y).to(device)
                        pad_yz = zeros_noise_scale * \
                            torch.randn(num_to_gen, ndim_tot - ndim_y - ndim_z)
                        y_samps = torch.cat([z.to(device),
                                             pad_yz.to(device),
                                             y_samps], dim=1)
                        y_samps = y_samps.to(device)
                    with torch.no_grad():
                        # Using the GLOW model
                        if 'Nflow' in self.CINN_obj:
                            X_pred = self.model(z, y_samps)[0].cpu()
                            H_pred = self.model.cinn(x_samps, y_samps)[0].cpu()
                        else:
                            full_inverse = self.model(y_samps, rev=True)[0]
                            X_pred = full_inverse[:, :ndim_x].cpu()
                            H_pred = self.model(x_samps)[0][:, :ndim_z].cpu()
                            # Y_pred = full_inverse[:, -ndim_y:].cpu()
                            # self.y_samps = y_samps[:, -ndim_y:].cpu().cpu().clone()
                            # self.X_pred = X_pred.clone()
                            # self.Y_pred = Y_pred.clone()
                            # self.H_pred = H_pred.clone()
                    if C > 1:
                        X_pred = X_pred.reshape(num_to_gen, V, C)
                    H = noise_z.reshape(num_to_gen, V, C).cpu()
                    H_pred = H_pred.reshape(num_to_gen, V, C)
                    # H, H_pred = None, None  # So figures are more compact
                viz.plt_generation_fig_competitor(
                    self, X_sub, X_pred, Y[which_rows], H, H_pred)
                train_test_save = '_train' if viz_train else '_test'
                save_prefix = self.path+'/' if self.prefix == '' else self.prefix
                self.fig_gen.savefig(f'{save_prefix}Generation{train_test_save}_epoch{self.epoch+1}_top{pp+1}_occurrences.png',
                                     dpi=150, bbox_inches='tight', pad_inches=0)
                self.final_viz = False
                # Also report quantitative metrics:
                if self.final_viz:
                    # Record num of obs.
                    X_sub, X_pred = X_sub.flatten(
                        start_dim=1), X_pred.flatten(start_dim=1)
                    self.two_sample_stat[Y_row] = [num_to_gen]
                    for method in ['MMD', 'Energy']:
                        if method == 'MMD':
                            for alphas in [[0.1], [1.0], [5.0], [10.0]]:
                                ret = two_sample_mtd(
                                    X_sub, X_pred, alphas=alphas, method=method)
                                self.two_sample_stat[Y_row].append(ret)
                        else:
                            ret = two_sample_mtd(X_sub, X_pred, method=method)
                            self.two_sample_stat[Y_row].append(ret)

    def load_from_competitors_checkpoint(self):
        if len(self.model) > 1:
            # CGAN
            self.path = f'{self.data_name}_CGAN'
            self.net_D, self.net_G = self.model
            self.optimizer_D, self.optimizer_G = self.optimizer
            isExist = os.path.exists(self.path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(self.path)
                print("The new directory is created!")
            self.checkpoint_savename = f'{self.path}/CGAN_checkpoint'
            if os.path.exists(self.checkpoint_savename) and self.resume_checkpoint:
                # Resume training if this file exist
                checkpoint = torch.load(self.checkpoint_savename, map_location=torch.device(
                    'cpu')) if self.cpu_load else torch.load(self.checkpoint_savename)
                self.net_D.load_state_dict(checkpoint['D_state_dict'])
                self.optimizer_D.load_state_dict(checkpoint['D_optimizer'])
                self.net_G.load_state_dict(checkpoint['G_state_dict'])
                self.optimizer_G.load_state_dict(checkpoint['G_optimizer'])
                self.epoch = checkpoint['epoch']
                self.loss_GAN_train, self.loss_GAN_test = checkpoint['loss_ls']
            else:
                self.epoch = 0
                self.loss_GAN_train, self.loss_GAN_test = [], []
        else:
            # CINN
            self.model = self.model[0]
            self.path = f'{self.data_name}_CINN'
            isExist = os.path.exists(self.path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(self.path)
                print("The new directory is created!")
            self.checkpoint_savename = f'{self.path}/CINN_checkpoint'
            if os.path.exists(self.checkpoint_savename) and self.resume_checkpoint:
                # Resume training if this file exist
                checkpoint = torch.load(self.checkpoint_savename, map_location=torch.device(
                    'cpu')) if self.cpu_load else torch.load(self.checkpoint_savename)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epoch = checkpoint['epoch']
                self.loss_CINN_train, self.loss_CINN_test = checkpoint['loss_ls']
            else:
                self.epoch = 0
                self.loss_CINN_train, self.loss_CINN_test = [], []

    def save_competitors_checkpoint(self):
        if 'CGAN' in self.path:
            checkpoint = {'epoch': self.epoch + 1, 'loss_ls': [self.loss_GAN_train, self.loss_GAN_test],
                          'D_state_dict': self.net_D.state_dict(), 'D_optimizer': self.optimizer_D.state_dict(),
                          'G_state_dict': self.net_G.state_dict(), 'G_optimizer': self.optimizer_G.state_dict()}
        if 'CINN' in self.path:
            checkpoint = {'epoch': self.epoch + 1, 'loss_ls': [self.loss_CINN_train, self.loss_CINN_test],
                          'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, self.checkpoint_savename)

    '''Visualizations of our method'''

    def viz_generation(self, viz_train=True):
        if viz_train:
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        Unique_Y, counts_Y = torch.unique(Y, return_counts=True, dim=0)
        counts_Y, idx = torch.sort(counts_Y, descending=True)
        Unique_Y = Unique_Y[idx]
        viz_freq = 5 if self.num_viz == 1 else 15
        # viz_freq = 1
        if self.epoch > 50:
            viz_freq = 10
        # if self.epoch == 0 or np.mod(self.epoch + 1, viz_freq) == 0 or self.final_viz:
        if self.final_viz:
            self.two_sample_stat = {}
        if np.mod(self.epoch + 1, viz_freq) == 0 or self.final_viz:
            for pp, Y_row in enumerate(Unique_Y[:self.num_viz]):
                if self.V == 1 and self.C == 2:
                    # Two moon plot
                    Y_row = None
                print(
                    f'Checking generation at \n {Y_row} with {counts_Y[pp]} out of {len(Y)} data')
                # pdb.set_trace()
                N, V, C = X.shape
                NV = int(N*V)
                H_full = torch.zeros(NV, C).to(device)
                for i, base_disti in enumerate(self.base_dist_ls):
                    torch.manual_seed(1103)
                    idx_i = Y == i
                    H_full[idx_i.flatten()] = base_disti.rsample(sample_shape=(
                        idx_i.sum().cpu().detach().numpy(),))
                H_full = H_full.reshape(N, V, C)
                if self.final_viz:
                    self.two_sample_mtd = two_sample_mtd
                # # Visualize results
                viz.visualize_generation_one_graph(
                    self, X, Y, H_full, Y_row)
                train_test_save = '_train' if viz_train else '_test'
                save_prefix = self.path+'/' if self.prefix == '' else self.prefix
                if self.C > 1:
                    self.fig_gen.savefig(f'{save_prefix}Generation{train_test_save}_epoch{self.epoch+1}_top{pp+1}_occurrences.png',
                                         dpi=150, bbox_inches='tight', pad_inches=0)
                if self.C == 1:
                    # Also visualize correlation matrix, so we save it here as well
                    save_prefix = self.path+'/' if self.prefix == '' else self.prefix
                    self.fig_corr.savefig(
                        f'{save_prefix}correlation_matrices_epoch{self.epoch+1}.png', dpi=100, bbox_inches='tight', pad_inches=0)
                if self.V == 1 and self.C == 2:
                    # No lonegr need to plot, because we have finished two-moon plot
                    break

    def get_GIF(self, Y_row, viz_train=True, from_X_to_H=False):
        self.from_X_to_H = from_X_to_H
        if self.C == 1:
            raise ValueError('GIF for C=1 not yet considered')
        if viz_train:
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_test, self.Y_test
        which_rows = (Y == Y_row).all(
            dim=1) if Y_row is not None else torch.tensor([True]).repeat(X.shape[0])
        N, V, C = X.shape
        NV = int(N*V)
        H_full = torch.zeros(NV, C).to(device)
        for i, base_disti in enumerate(self.base_dist_ls):
            torch.manual_seed(1103)
            idx_i = Y == i
            H_full[idx_i.flatten()] = base_disti.rsample(sample_shape=(
                idx_i.sum().cpu().detach().numpy(),))
        H_full = H_full.reshape(N, V, C)
        X, Y, H_full = X[which_rows], Y[which_rows], H_full[which_rows]
        num_plot = min(6000, X.shape[0])
        self.transport_cost_XtoH_ls = []
        self.transport_cost_HtoX_ls = []
        viz.save_trajectory_revised(
            self, X[:num_plot], Y[:num_plot], H_full[:num_plot])

    def viz_losses(self):
        if np.mod(self.epoch + 1, 1) == 0 or self.final_viz:
            fig = viz.losses_and_error_plt_real_data_on_graph(self)
            save_prefix = self.path+'/' if self.prefix == '' else self.prefix
            fig.savefig(f'{save_prefix}Losses_epoch{self.epoch+1}.png',
                        dpi=150, bbox_inches='tight', pad_inches=0)

    '''Saving and/or loading from checkpoints for our method'''

    def save_checkpoint(self):
        # Save checkpoint at the current epoch, so we can resume training etc
        self.loss_ls = [self.loss_g_ls_train, self.loss_g_ls_test, self.loss_c_ls_train,
                        self.loss_c_ls_test, self.classify_error_ls_train, self.classify_error_ls_test, self.loss_w2_ls_train, self.loss_w2_ls_test]
        checkpoint = {'epoch': self.epoch + 1, 'loss_ls': self.loss_ls,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'optimizer_gen_net': self.optimizer_gen_net.state_dict(),
                      'gen_net': self.gen_net.state_dict()}
        # if len(self.non_invertible_ls) > 0 and self.non_invertible_ls[-1] == self.epoch:
        #     # Rename the latest saved model, as its NEXT epoch (which is the current epoch) has non-invertibility issue
        #     # torch.save(
        #     #     checkpoint, f'{self.checkpoint_savename}_non_invertible_at_epoch{self.epoch}')
        #     # os.rename(self.checkpoint_savename, f'{self.checkpoint_savename}_non_invertible_next_epoch{self.epoch}')
        #     shutil.copy(self.checkpoint_savename,
        #                 f'{self.checkpoint_savename}_non_invertible_next_epoch{self.epoch}')
        # else:
        #     torch.save(checkpoint, self.checkpoint_savename)
        torch.save(checkpoint, self.checkpoint_savename)

    def load_from_checkpoint(self):
        # Check whether the specified folder exists or not
        self.path = self.data_name
        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)
            print("The new directory is created!")
        # Load from a previous training iteration
        self.checkpoint_savename = f'{self.path}/IResNet_checkpoint_dim_{self.dim}_nblocks_{self.nblocks}'
        if os.path.exists(self.checkpoint_savename) and self.resume_checkpoint:
            # Resume training if this file exist
            checkpoint = torch.load(self.checkpoint_savename, map_location=torch.device(
                'cpu')) if self.cpu_load else torch.load(self.checkpoint_savename)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.gen_net.load_state_dict(checkpoint['gen_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optimizer_gen_net.load_state_dict(
                checkpoint['optimizer_gen_net'])
            self.epoch = checkpoint['epoch']
            if self.load_losses:
                self.loss_g_ls_train, self.loss_g_ls_test, self.loss_c_ls_train, self.loss_c_ls_test, self.classify_error_ls_train, self.classify_error_ls_test, self.loss_w2_ls_train, self.loss_w2_ls_test = checkpoint[
                    'loss_ls']
        else:
            self.epoch = 0
            self.loss_g_ls_train, self.loss_g_ls_test, self.loss_c_ls_train, self.loss_c_ls_test, self.classify_error_ls_train, self.classify_error_ls_test = [], [], [], [], [], []
            self.loss_w2_ls_train, self.loss_w2_ls_test = [], []

    '''Other necessary helpers for our method'''

    def get_H_cond_Y(self):
        if '8_gaussian' in self.path:
            base_mu0 = self.gen_net(torch.Tensor(
                [1, 0, 0, 0]).to(device)).to(device)
            base_mu1 = self.gen_net(torch.Tensor(
                [0, 1, 0, 0]).to(device)).to(device)
            base_mu2 = self.gen_net(torch.Tensor(
                [0, 0, 1, 0]).to(device)).to(device)
            base_mu3 = self.gen_net(torch.Tensor(
                [0, 0, 0, 1]).to(device)).to(device)
            dist = 0.1
            base_cov = (torch.eye(self.gen_net.fc.out_features)
                        * dist).to(device)
            self.base_dist0 = MultivariateNormal(base_mu0, base_cov)
            self.base_dist1 = MultivariateNormal(base_mu1, base_cov)
            self.base_dist2 = MultivariateNormal(base_mu2, base_cov)
            self.base_dist3 = MultivariateNormal(base_mu3, base_cov)
            self.base_dist_ls = [
                self.base_dist0, self.base_dist1, self.base_dist2, self.base_dist3]
        elif 'three_moon' in self.path:
            base_mu0 = self.gen_net(torch.Tensor(
                [1, 0, 0]).to(device)).to(device)
            base_mu1 = self.gen_net(torch.Tensor(
                [0, 1, 0]).to(device)).to(device)
            base_mu2 = self.gen_net(torch.Tensor(
                [0, 0, 1]).to(device)).to(device)
            dist = 0.05
            base_cov = (torch.eye(self.gen_net.fc.out_features)
                        * dist).to(device)
            self.base_dist0 = MultivariateNormal(base_mu0, base_cov)
            self.base_dist1 = MultivariateNormal(base_mu1, base_cov)
            self.base_dist2 = MultivariateNormal(base_mu2, base_cov)
            self.base_dist_ls = [self.base_dist0,
                                 self.base_dist1, self.base_dist2]
        else:
            in0 = torch.Tensor([1, 0]).to(device)
            in1 = torch.Tensor([0, 1]).to(device)
            base_mu0 = self.gen_net(in0).to(device)
            base_mu1 = self.gen_net(in1).to(device)
            # NOTE, should instead fix the covariance to be 1.
            dist = 0.1 if self.X_test is None else torch.linalg.norm(
                base_mu0 - base_mu1).cpu() / 8  # First for 3 node simulation, Second for real-data
            if self.X_dist == 'many_node_graph':
                # Large cond. gen. graph, cov is identity
                dist = 1.
            base_cov = (torch.eye(self.gen_net.fc.out_features)
                        * dist).to(device)
            if self.C > 1:
                self.base_dist0 = MultivariateNormal(base_mu0, base_cov)
                self.base_dist1 = MultivariateNormal(base_mu1, base_cov)
            else:
                base_cov = torch.ones(1).to(device)
                self.base_dist0 = Normal(base_mu0, base_cov)
                self.base_dist1 = Normal(base_mu1, base_cov)
            self.base_dist_ls = [self.base_dist0, self.base_dist1]
        self.sigma_HY = np.sqrt(dist)

    def check_model_inversion(self):
        # On-dist inversion error
        N_sub = 10
        # Randomly sample some training indices
        with torch.no_grad():
            logdet = False
            X_rand = self.X_train[np.random.choice(np.arange(self.N), N_sub)]
            X_for = self.model(
                X_rand.flatten(start_dim=1), self.edge_index, self.edge_weight, logdet=logdet)
            X_for = X_for.reshape(N_sub, self.V, self.C).to(device)
            X_hat = self.model.inverse(
                X_for, self.edge_index, self.edge_weight)
            err_percent_on_dist = torch.linalg.norm(
                X_hat-X_rand)/torch.linalg.norm(X_rand)
            if self.X_test is not None:
                X_rand = self.X_test[np.random.choice(
                    np.arange(self.X_test.shape[0]), N_sub)]
                X_for = self.model(
                    X_rand.flatten(start_dim=1), self.edge_index, self.edge_weight, logdet=logdet)
                X_for = X_for.reshape(N_sub, self.V, self.C).to(device)
                X_hat = self.model.inverse(
                    X_for, self.edge_index, self.edge_weight)
                err_percent_on_dist_test = torch.linalg.norm(
                    X_hat-X_rand)/torch.linalg.norm(X_rand)
            # Off-dist. inversion error
            X = torch.rand(N_sub, self.V, self.C).to(device)
            X_for = self.model(
                X.flatten(start_dim=1), self.edge_index, self.edge_weight, logdet=logdet)
            X_for = X_for.reshape(N_sub, self.V, self.C).to(device)
            X_hat = self.model.inverse(
                X_for, self.edge_index, self.edge_weight)
            err_percent_off_dist = torch.linalg.norm(
                X_hat-X)/torch.linalg.norm(X)
        if self.X_test is not None:
            if err_percent_on_dist_test > self.err_on_dist:
                self.err_on_dist = err_percent_on_dist_test
        else:
            if err_percent_on_dist > self.err_on_dist:
                self.err_on_dist = err_percent_on_dist
        if err_percent_off_dist > self.err_off_dist:
            self.err_off_dist = err_percent_off_dist
        if err_percent_off_dist < 1e-4 and err_percent_on_dist < 1e-4:
            print(
                f'Model is invertible \n On Dist. Inversion Error is {err_percent_on_dist*100}% \n Off Dist. Inversion Error is {err_percent_off_dist*100}%')
            if self.X_test is not None:
                print(
                    f'On Dist. Inversion Error on X_test is {err_percent_on_dist_test*100}%')
            print(f'Past non-invertible epochs are {self.non_invertible_ls}')
        else:
            print(
                f'Model non-invertible \n On Dist. Inversion Error is {err_percent_on_dist*100}% \n Off Dist. Inversion Error is {err_percent_off_dist*100}%')
            if self.X_test is not None:
                print(
                    f'On Dist. Inversion Error on X_test is {err_percent_on_dist_test*100}%')
            self.non_invertible_ls.append(self.epoch)
            self.save_checkpoint()
            print(f'Weight Reduction in place')
            self.model.small_weights()

# For CINN, MMD


def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)


def fit(input, target):
    return torch.mean((input - target)**2)


def mem_report():
    if device.type == 'cuda':
        GPUs = GPUtil.getGPUs()
        for i, gpu in enumerate(GPUs):
            print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(
                i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
    else:
        print("CPU RAM Free: "
              + humanize.naturalsize(psutil.virtual_memory().available))

# Two-sample tests:


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2)
                 + norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum()
               + self.a00 * (k_1.sum() - torch.trace(k_1))
               + self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd


class EnergyStatistic:
    r"""The energy test of :cite:`szekely2013energy`.

    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = - 1. / (n_1 * n_1)
        self.a11 = - 1. / (n_2 * n_2)
        self.a01 = 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, ret_matrix=False):
        r"""Evaluate the statistic.

        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        norm : float
            Which norm to use when computing distances.
        ret_matrix: bool
            If set, the call with also return a second variable.

            This variable can be then used to compute a p-value using
            :py:meth:`~.EnergyStatistic.pval`.

        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)
        d_1 = distances[:self.n_1, :self.n_1].sum()
        d_2 = distances[-self.n_2:, -self.n_2:].sum()
        d_12 = distances[:self.n_1, -self.n_2:].sum()

        loss = 2 * self.a01 * d_12 + self.a00 * d_1 + self.a11 * d_2

        if ret_matrix:
            return loss, distances
        else:
            return loss


def two_sample_mtd(x, y, alphas=[1.0], method='MMD'):
    """
        Return the statistics based on input method
    """
    if x.device.type != 'cpu':
        x, y = x.cpu().detach(), y.cpu().detach()
    torch.manual_seed(1103)
    indexes = torch.randperm(x.shape[0])
    torch.manual_seed(1111)
    indexes1 = torch.randperm(y.shape[0])
    x, y = x[indexes], y[indexes1]
    cuda = True if device.type != 'cpu' else False
    N_1, N_2 = x.shape[0], y.shape[0]
    if method == 'MMD':
        mtd = MMDStatistic(N_1, N_2)
        return mtd(x, y, alphas).item()
    if method == 'Energy':
        mtd = EnergyStatistic(N_1, N_2)
        return mtd(x, y).item()

# Other helpers


def get_stat_frame_from_dict(res_dict):
    '''
        res_dict: {generative method: {unique_Y: [counts, two_sample_stats]}}
    '''
    keys_gen_mtd = list(res_dict.keys())
    keys_uniq_Y = list(res_dict[keys_gen_mtd[0]].keys())
    nrow, ncol = len(keys_gen_mtd), len(
        res_dict[keys_gen_mtd[0]][keys_uniq_Y[0]])-1  # Num generative method & Num two-sample tests
    l_Y = len(keys_uniq_Y)
    ncol_full = ncol*l_Y
    res_array = np.zeros((nrow, ncol))
    res_array_full = np.zeros((nrow, ncol_full))
    counts = np.array([res_dict[keys_gen_mtd[0]][key][0]
                      for key in keys_uniq_Y])
    weights = counts/counts.sum()
    for i in range(nrow):
        keys_uniq_Y = list(res_dict[keys_gen_mtd[i]].keys())
        for j in range(ncol):
            vals = np.array([res_dict[keys_gen_mtd[i]][key][j+1]
                            for key in keys_uniq_Y])
            res_array[i, j] = weights.dot(vals)
            res_array_full[i, j*l_Y:(j+1)*l_Y] = vals
    cols = [f'MMD: alpha={i}' for i in [0.1, 1.0, 5.0, 10.0]]+['Energy']
    if l_Y > 1:
        keys_uniq_Y = [
            str(i.cpu())+f' {j} obs' for i, j in zip(keys_uniq_Y, counts)]
    else:
        keys_uniq_Y = [str(i)+f' {j} obs' for i, j in zip(keys_uniq_Y, counts)]
    cols_Y = np.repeat(keys_uniq_Y, ncol, axis=0)
    cols_full = list(zip(*[cols_Y, np.tile(cols, l_Y)]))
    cols_full = pd.MultiIndex.from_tuples(
        cols_full, names=["Ys", "two_sample"])
    res_array = pd.DataFrame(
        res_array, index=keys_gen_mtd, columns=cols).round(3)
    res_array_full = pd.DataFrame(
        res_array_full, index=keys_gen_mtd, columns=cols_full).round(3)
    res_array_full.index.name = 'gen_method'
    return [res_array, res_array_full.T]

###################

###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
