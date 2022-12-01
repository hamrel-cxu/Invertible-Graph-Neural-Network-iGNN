from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import matplotlib.cm as cm
import os
from scipy.stats import kde
from moviepy.editor import *
from matplotlib.ticker import MaxNLocator
# from pdf2image import convert_from_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plt_X_at_Y(X, Y, G, self=None, Hs=None):
    Unique_Y, counts_Y = torch.unique(Y, return_counts=True, dim=0)
    counts_Y, idx = torch.sort(counts_Y, descending=True)
    Y_row = Unique_Y[0]  # Just check at the first Y
    print(f'Plot at Y={Y_row}')
    which_rows = (Y == Y_row).all(dim=1)
    X = X[which_rows]
    if self is not None:
        with torch.no_grad():
            H = Hs[which_rows]
            Xest = self.model.inverse(H, self.edge_index, maxIter=50).cpu()
            # keep = (Xest.norm(dim=1) < 100).all(dim=1) # This is wrong
            Hest = self.model.forward(X.flatten(
                start_dim=1), self.edge_index, logdet=False).cpu()
            Hest = Hest.reshape(X.shape)
            # X = X[keep]
            # Xest = Xest[keep]
            # H = H[keep]
            # Hest = Hest[keep]
    np.random.seed(1103)
    nodes = np.random.choice(list(G.nodes), 5, replace=False)
    for ref_node in nodes:
        neighbors = np.unique(np.array([list(i)
                              for i in G.edges(ref_node)]), axis=0)[:, 1]
        neighbors = neighbors[neighbors != ref_node]
        nodes_to_plt = torch.from_numpy(
            np.append(neighbors, ref_node)[::-1].copy())
        nodes_to_plt = nodes_to_plt.int().cpu().detach().numpy()
        if self is not None:
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        else:
            fig, ax = plt.subplots(figsize=(4, 4))
        Ys = np.linspace(0, 1, len(nodes_to_plt))
        Ys[Ys == 0.5] = 0.45
        for i, node in enumerate(nodes_to_plt):
            color = np.array([cm.seismic(Ys[i])])
            Xs = X[:, node, :].cpu().detach().numpy()
            xlabel = f'{node}: Y={Y_row[node]}'
            if self is not None:
                Xests = Xest[:, node, :].cpu().detach().numpy()
                Hs = H[:, node, :].cpu().detach().numpy()
                Hests = Hest[:, node, :].cpu().detach().numpy()
                ax[0].scatter(Xs[:, 0], Xs[:, 1], c=color,
                              label=xlabel)
                ax[0].set_title(r'$X|Y$', fontsize=20)
                ax[1].scatter(Xests[:, 0], Xests[:, 1], c=color)
                ax[1].set_title(r'$\hat{X}|Y$', fontsize=20)
                ax[1].get_shared_x_axes().join(ax[1], ax[0])
                ax[1].get_shared_y_axes().join(ax[1], ax[0])
                ax[2].scatter(Hs[:, 0], Hs[:, 1], c=color)
                ax[2].set_title(r'$H|Y$', fontsize=20)
                ax[3].scatter(Hests[:, 0], Hests[:, 1], c=color)
                ax[3].set_title(r'$\hat{H}|Y$', fontsize=20)
                ax[3].get_shared_x_axes().join(ax[3], ax[2])
                ax[3].get_shared_y_axes().join(ax[3], ax[2])
            else:
                ax.scatter(Xs[:, 0], Xs[:, 1], c=color,
                           label=xlabel)
                ax.set_title(r'$X|Y$', fontsize=20)
        if self is not None:
            ax[0].legend(loc='upper center', ncol=2, fontsize=16)
        else:
            ax.legend(loc='upper center', ncol=2, fontsize=16)
        fig.suptitle(
            f'Plot at community of nodes {nodes_to_plt}', fontsize=20, y=1.05)
        if self is not None:
            self.fig_gen = fig
        plt.show()
        plt.close()


def plot_contour_over_region(X_pred, H_full, H_val, type='log_prob', dataname='two_moon', savefig=True):
    '''
        H_full: the set of H|Y over all Y
        H_val: some metric based on H_full.
        For instance:
            -> type = 'log_prob':
               H_val = H_full_log_prob: evaluate H on its H|Y (do so when generating H|Y)
                so we get confidence region

            -> type = 'classify_prob'
               H_val = H_full_prob: predict H based on linear classifier
        X_pred = F^{-1}(H_full)
    '''
    import matplotlib.colors
    # Contour of H, based on value of Z
    xX, yX = X_pred[:, 0].cpu().detach().numpy(), X_pred[:,
                                                         1].cpu().detach().numpy()
    xH, yH = H_full[:, 0].cpu().detach().numpy(), H_full[:,
                                                         1].cpu().detach().numpy()
    z = H_val.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    if type == 'log_prob':
        quantiles = np.arange(10, 41, 5)
        levels = np.unique(np.percentile(z, q=quantiles))
        true_quantiles = 100-quantiles[::-1]
        print(f'Quantiles: {true_quantiles} \n Log_prob: {levels}')
    if type == 'classify_prob':
        levels = np.round(np.linspace(0.01, 0.99, 11), 2)
    lwidth = 1
    cmap = 'seismic'
    ax[0].tricontour(xX, yX, z, levels, linewidths=lwidth, cmap=cmap)
    cax = ax[1].tricontour(xH, yH, z, levels, linewidths=lwidth, cmap=cmap)
    ax[0].set_title(r'$\hat{X}|Y$', fontsize=18)
    ax[1].set_title(r'$H|Y$', fontsize=18)
    # if type == 'log_prob':
    #     fig.suptitle('Confidence region contour plot', y=1.05, fontsize=18)
    # if type == 'classify_prob':
    #     fig.suptitle(r'Contour plot based on $P(Y=1|H)$', y=1.05, fontsize=18)
    norm = matplotlib.colors.Normalize(
        vmin=cax.cvalues.min(), vmax=cax.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cax.cmap)
    cbar = fig.colorbar(sm)
    if type == 'log_prob':
        ylab = 'Confidence level'
        suff_save = 'confidence'
        cbar.ax.set_yticklabels(true_quantiles[::-1], fontsize=18)
    if type == 'classify_prob':
        ylab = 'P(Y=1|H)'
        suff_save = 'prob1'
        # Force this many ticklabels
        cbar.ax.set_ylim([0, 1])
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(len(levels)-1))
        levels = np.round(np.linspace(0, 1, 11), 2)
        cbar.ax.set_yticklabels(levels[::-1], fontsize=16)
    for a in ax.ravel():
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])
    cbar.ax.set_ylabel(ylab, rotation=270, labelpad=15, fontsize=18)
    fig.savefig(f'{dataname}_{suff_save}_contour.png',
                dpi=200, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    return fig


def plot_img_over_trajectory(input_all_ls, gamma_ls, num_per_row, from_X_to_H=False, Ys=None):
    '''
        input_all_ls: each entry has num_tot_blocks+1 X N X dimension
        num_per_row: how many we plot out of num_tot_blocks
    '''
    fig, ax = plt.subplots(len(gamma_ls), num_per_row, figsize=(4*num_per_row, 4*len(gamma_ls)),
                           constrained_layout=True, sharex=True, sharey=True)
    for i, gamma in enumerate(gamma_ls):
        input_all = input_all_ls[i]
        num_tot_blocks = input_all.shape[0]
        selected_blocks = torch.linspace(
            0, num_tot_blocks-1, num_per_row, dtype=torch.int)
        fsize = 32
        for j in range(num_per_row):
            block_id = selected_blocks[j]
            data = input_all[block_id]
            x = data[:, 0]
            y = data[:, 1]
            if len(gamma_ls) > 1:
                a_now = ax[i, j]
            else:
                a_now = ax[j]
            if Ys is None:
                # Plot density map
                val = 3.5
                xmin, xmax, ymin, ymax = -val, val, -val, val
                heatmap, xedges, yedges = np.histogram2d(
                    x, y, range=[[xmin, xmax], [ymin, ymax]], bins=256)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                a_now.imshow(heatmap.T, extent=extent,
                             cmap='viridis', origin='lower', aspect="auto")
            else:
                a_now.scatter(x, y, c=Ys, s=2)
            a_now.set_xlabel(f'Block {block_id}', fontsize=fsize)
        if len(gamma_ls) > 1:
            a_now = ax[i, 0]
        else:
            a_now = ax[0]
        # a_now.set_ylabel(f'$\gamma$={gamma}', fontsize=28)
        if len(gamma_ls) > 1:
            a_now = ax[0, 0]
            a_now1 = ax[0, -1]
        else:
            a_now = ax[0]
            a_now1 = ax[-1]
        if from_X_to_H:
            a_now.set_title(r'$X|Y$', fontsize=fsize)
            a_now1.set_title(r'$\hat{H}|Y$', fontsize=fsize)
        else:
            a_now.set_title(r'$H|Y$', fontsize=fsize)
            a_now1.set_title(r'$\hat{X}|Y$', fontsize=fsize)
    for a in ax.ravel():
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])
    return fig


def visualize_generation_one_graph(self, X, Y, H_full, Y_row=None):
    '''
        self: the object which has many methods, including computing L_g
        Y_row: a specific choice of Y such that we only show inverse at this Y
        If None, plot everything
    '''
    # For two moon, after training the models
    # Basically visualize how the original density is gradually transformed to the data density
    # NOTE: due to speed in inversion, we just examine result over a subset of total data
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['figure.titlesize'] = 24
    if self.X_dist == 'many_node_graph':
        plt_X_at_Y(X, Y, self.G, self, H_full)
    else:
        which_rows = (Y == Y_row).all(
            dim=1) if Y_row is not None else torch.tensor([True]).repeat(X.shape[0])
        ########################################################################
        # with torch.no_grad():
        #     # Somehow much have it moved to cpu
        batch_idx = np.arange(X.shape[0])[which_rows.cpu()]
        self.viz = True
        L_g_now, _, _ = self.get_L_g(batch_idx, X, Y)
        L_g_now = np.around(L_g_now.item(), 2)
        self.viz = False
        ########################################################################
        X, Y, H_full = X[which_rows], Y[which_rows], H_full[which_rows]
        start = timer()
        maxIter = 200 if self.final_viz else 50
        X_pred = self.model.inverse(
            H_full, self.edge_index, maxIter=maxIter).cpu()
        # For some dataset, need to remove "outlier"
        # like the 2 circle one
        keep = (X_pred.norm(dim=1) < 100).all(dim=1)
        X = X[keep]
        X_pred = X_pred[keep]
        H_full = H_full[keep]
        Y = Y[keep]
        H_full = H_full.cpu().detach()
        N = X.shape[0]
        print(f'Invert {N} samples took {timer()-start} secs')
        with torch.no_grad():
            H_pred = self.model.forward(X.flatten(
                start_dim=1), self.edge_index, logdet=False).cpu()
        H_pred = H_pred.reshape(X.shape)
        X = X.cpu()
        # # Visualize X and Inverse of H
        # num_to_plot = 1000 if '8_gaussian' in self.path else 600
        # if self.V > 1 or self.C > 2:
        #     num_to_plot = 100
        num_to_plot = N
        if self.C > 1:
            # Scatter plot
            plt_generation_fig(self, X[:num_to_plot], X_pred[:num_to_plot],
                               Y[:num_to_plot], H_full[:num_to_plot], H_pred[:num_to_plot], L_g_now)
        if self.C == 1:
            # Graph GP, so we want to visualize the covariances
            plot_and_save_corr(self, X, X_pred, H_full, H_pred)
        # Also report quantitative metrics:
        if self.final_viz:
            # Record num of obs.
            X_sub, X_pred = X.flatten(
                start_dim=1), X_pred.flatten(start_dim=1)
            self.two_sample_stat[Y_row] = [N]
            for method in ['MMD', 'Energy']:
                if method == 'MMD':
                    for alphas in [[0.1], [1.0], [5.0], [10.0]]:
                        ret = self.two_sample_mtd(
                            X_sub, X_pred, alphas=alphas, method=method)
                        self.two_sample_stat[Y_row].append(ret)
                else:
                    ret = self.two_sample_mtd(X_sub, X_pred, method=method)
                    self.two_sample_stat[Y_row].append(ret)


def save_trajectory_revised(self, X, Y, H_full):
    '''
        NOTE: Here X can either be true data sample OR the base sample
        Then the def. of H_full also changes
        If X are base samples, then we must use forward mapping of blocks
    '''
    V_tmp = X.shape[1]
    savedir = f'{self.path}'
    fsize = 22
    N = X.shape[0]
    blocks = self.model.blocks if self.from_X_to_H else reversed(
        self.model.blocks)
    X_np, H_np = X.flatten(start_dim=0, end_dim=1), H_full.flatten(
        start_dim=0, end_dim=1)
    if self.C > 2:
        V_tmp = int(self.C/2)
        C_tmp = 2
        X_np = X.reshape(N, V_tmp, C_tmp).flatten(start_dim=0, end_dim=1)
        H_np = H_full.reshape(N, V_tmp, C_tmp).flatten(start_dim=0, end_dim=1)
    xmin, xmax = min(X_np[:, 0].min(), H_np[:, 0].min()).item(), max(
        X_np[:, 0].max(), H_np[:, 0].max()).item()
    ymin, ymax = min(X_np[:, 1].min(), H_np[:, 1].min()).item(), max(
        X_np[:, 1].max(), H_np[:, 1].max()).item()
    same_row = self.same_row
    with torch.no_grad():
        t = 0
        # Gradually invert H_full through each layer to see how it matches the original density
        for block in blocks:
            block.logdet = False
            if self.from_X_to_H:
                # Here H_full is actually X
                X_pred, Fx, _ = block(
                    X.flatten(start_dim=1), self.edge_index, self.edge_weight) if self.edge_index is not None else block(X.flatten(start_dim=1))
                transport_cost = (torch.linalg.norm(Fx.flatten(start_dim=1),
                                                    dim=1)**2/2).sum().item()/N
                X_pred = X_pred.reshape(X.shape)
                self.transport_cost_XtoH_ls.append(transport_cost)
            else:
                if self.C > 2:
                    H_full = H_full.flatten(start_dim=1)
                X_pred = block.inverse(
                    H_full, self.edge_index, self.edge_weight) if self.edge_index is not None else block.inverse(H_full)
                # For some dataset, need to remove "outlier"
                # like the 2 circle one
                keep = (X_pred.norm(dim=1) < 100).all(dim=1)
                X = X[keep]
                X_pred = X_pred[keep]
                H_full = H_full[keep]
                Y = Y[keep]
                N = X_pred.shape[0]
                ###############
                transport_cost = (torch.linalg.norm(
                    (X_pred-H_full).flatten(start_dim=1), dim=1)**2/2).sum().item()/N
                self.transport_cost_HtoX_ls.append(transport_cost)
            if self.C > 2:
                V_tmp = int(self.C/2)
                C_tmp = 2
                X = X.reshape(N, V_tmp, C_tmp)
                X_pred = X_pred.reshape(N, V_tmp, C_tmp)
                H_full = H_full.reshape(N, V_tmp, C_tmp)
            if self.from_X_to_H:
                transport_cost_ls = self.transport_cost_XtoH_ls
            else:
                transport_cost_ls = self.transport_cost_HtoX_ls
            # Include transport cost on the top
            fig = plt.figure(figsize=(8, 11))
            spec = fig.add_gridspec(5, 2)
            if same_row:
                fig = plt.figure(figsize=(16, 4))
                spec = fig.add_gridspec(1, 4)
            # Plot transport cost
            if same_row == False:
                ax = fig.add_subplot(spec[0, :])
                ax.plot(transport_cost_ls, '-o')
                # ax.set_xlabel('Block')
                if self.from_X_to_H:
                    ax.set_title(
                        r'$W_2$ transport cost of $X \rightarrow H$ over blocks')
                else:
                    ax.set_title(
                        r'$W_2$ transport cost of $H \rightarrow X$ over blocks')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_facecolor('lightblue')
            # colors = np.tile(cm.rainbow(np.linspace(0, 1, V_tmp)), (N, 1))
            colors = Y.cpu().detach().numpy()
            # if V_tmp == 1:
            #     # Two-moon or 8_gaussian
            #     if '8_gaussian' in self.path:
            #         colors = np.repeat('r', N)
            #         colors[(Y == 1).cpu().detach().numpy().flatten()] = 'm'
            #         colors[(Y == 2).cpu().detach().numpy().flatten()] = 'y'
            #         colors[(Y == 3).cpu().detach().numpy().flatten()] = 'k'
            #     else:
            #         colors = np.repeat('black', N)
            #         colors[(Y == 1).cpu().detach().numpy().flatten()] = 'blue'
            plt_dict = {0: X, 1: X_pred}
            if self.from_X_to_H:
                plt_dict[0] = H_full
            # Plot target and estimates
            for j in range(2):
                if same_row:
                    ax = fig.add_subplot(
                        spec[0]) if j == 0 else fig.add_subplot(spec[1])
                else:
                    ax = fig.add_subplot(
                        spec[1:3, 0]) if j == 0 else fig.add_subplot(spec[1:3, 1])
                ax.set_facecolor('lightblue')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                if j == 0:
                    ax.xaxis.set_visible(False)
                if self.from_X_to_H:
                    title = r'Targets $H$' if j == 0 else r'Estimates $\hat{H}$'
                else:
                    title = r'Targets $X$' if j == 0 else r'Estimates $\hat{X}$'
                XorXpred = plt_dict[j]
                XorXpred_tmp = XorXpred.flatten(
                    start_dim=0, end_dim=1).cpu().numpy()
                if self.V > 1 or (self.V == 1 and self.C > 2):
                    ax.plot(XorXpred_tmp[:, 0], XorXpred_tmp[:, 1],
                            linestyle='dashed', linewidth=0.075)
                ax.scatter(XorXpred_tmp[:, 0],
                           XorXpred_tmp[:, 1], c=colors, s=2)
                if same_row == False:
                    ax.set_title(title, fontsize=fsize)
            # plot the density
            # # Not including transport cost on the top
            if same_row:
                ax = fig.add_subplot(spec[2])
            else:
                ax = fig.add_subplot(spec[3:, 0])
            ax.set_facecolor('lightblue')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            X_pred_tmp = X_pred.flatten(start_dim=0, end_dim=1).cpu().numpy()
            # Try to get density overlaid but different colors, since I have multiple blobs
            x, y = X_pred_tmp[:, 0], X_pred_tmp[:, 1]
            xy = np.vstack([x, y])
            k = kde.gaussian_kde([x, y])(xy)
            ax.scatter(x, y, c=k, s=2)
            if self.from_X_to_H:
                title = r"Density of $\hat{H}$"
            else:
                title = r"Density of $\hat{X}$"
            if same_row == False:
                ax.set_title(title, fontsize=fsize)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            # plot the vector field
            # # Not including transport cost on the top
            if same_row:
                ax = fig.add_subplot(spec[3])
            else:
                ax = fig.add_subplot(spec[3:, 1])
            ax.set_facecolor('lightblue')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            X_pred_pre = X if self.from_X_to_H else H_full
            X_pred_pre = X_pred_pre.flatten(
                start_dim=0, end_dim=1).cpu().numpy()
            directions = X_pred_tmp - X_pred_pre
            logmag = 2 * \
                np.log(np.hypot(directions[:, 0], directions[:, 1]))
            # Smaller scale = larger arrow
            ax.quiver(
                x, y, directions[:, 0], directions[:, 1],
                np.exp(logmag), cmap="coolwarm", scale=3.5, width=0.015, pivot="mid")
            if same_row == False:
                ax.set_title("Vector Field", fontsize=fsize)
            ax.yaxis.set_visible(False)
            fig.tight_layout()
            plt.savefig(os.path.join(
                savedir, f"viz-{t:05d}.jpg"))
            plt.show()
            t += 1  # For plot saving
            # Update H or X as input for next plot
            # Must place here, as o/w vector field not computed correctly
            if self.from_X_to_H:
                X = X_pred.clone()
            else:
                H_full = X_pred.clone()


def trajectory_to_gif(self):
    import subprocess
    savedir = f'{self.path}'
    # Smaller framerate reduces picture speed (desirable if num blocks small)
    # 10 for 40 blocks was pretty fast
    suffix = '_XtoH' if self.from_X_to_H else '_HtoX'
    out_path = os.path.join(savedir, f'trajectory_epoch{self.epoch}{suffix}')
    bashCommand = 'ffmpeg -framerate 5 -y -i {} {}'.format(os.path.join(
        savedir, 'viz-%05d.jpg'), out_path+'.mp4')
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    clip = (VideoFileClip(out_path+'.mp4'))
    clip.write_gif(out_path + '.gif')


def plot_and_save_corr(self, X, X_pred, H_full, H_pred):
    S_X_true = get_corrcoef(X)
    S_X_est = get_corrcoef(X_pred)
    S_H_true = get_corrcoef(H_full)
    S_H_est = get_corrcoef(H_pred)
    self.S_X_true = S_X_true
    self.S_X_est = S_X_est
    self.S_H_true = S_H_true
    self.S_H_est = S_H_est
    if X.shape[1] > 10:
        cmap = cm.get_cmap('seismic')
    else:
        cmap = cm.get_cmap('viridis')
    vmin, vmax = -1, 1  # To ensure same range for colormap
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    ax[0, 0].matshow(S_X_true, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0, 0].set_title(r'Correlation of $X$')
    ax[0, 1].matshow(S_X_est, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0, 1].set_title(r'Correlation of $\hat{X}$')
    ax[0, 2].matshow(S_X_true-S_X_est, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0, 2].set_title(r'Diff. of Correlation in $X$')
    ax[1, 0].matshow(S_H_true, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1, 0].set_title(r'Correlation of $H$')
    ax[1, 1].matshow(S_H_est, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1, 1].set_title(r'Correlation of $\hat{H}$')
    c = ax[1, 2].matshow(S_H_true-S_H_est, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1, 2].set_title(r'Diff. of Correlation in $H$')
    for a in ax.ravel():
        a.xaxis.set_ticks_position('bottom')
    cbar_ax = fig.add_axes([-0.17, 0.03, 0.1, 0.9])
    plt.colorbar(c, cax=cbar_ax)
    self.fig_corr = fig


def get_corrcoef(input):
    if len(input.shape) < 2:
        raise ValueError('Inpit must be at least 2 dimensional.')
    if len(input.shape) > 2:
        return torch.corrcoef(input[:, :, 0].T).cpu().detach().numpy()
    else:
        return torch.corrcoef(input.T).cpu().detach().numpy()


def plt_generation_fig(self, X, X_pred, Y, H_full, H_pred, L_g_now):
    plt_dict = {0: X, 1: X_pred, 2: H_full, 3: H_pred}
    V_tmp = X.shape[1]
    N = X.shape[0]
    if self.C > 2:
        # NOTE: this is because FC treated graph example in R^V-x-C as a vector in \R^V-by-C, so that we need reshaping for visualization
        V_tmp = int(self.C/2)
        C_tmp = 2
        for key in plt_dict.keys():
            plt_dict[key] = plt_dict[key].reshape(N, V_tmp, C_tmp)
    if self.final_viz and self.plot_sub:
        title_dict = {
            0: r'$X|Y$', 1: r'$\hat{X}|Y=F^{-1}(H|Y)$'}
        fig, axs = plt.subplots(1, 2, figsize=(2 * 4, 4))
    else:
        title_dict = {
            0: r'$X|Y$', 1: r'$\hat{X}|Y=F^{-1}(H|Y)$', 2: r'$H|Y$', 3: r'$\hat{H}|Y=F(X|Y)$'}
        fig, axs = plt.subplots(1, 4, figsize=(4 * 4, 4))
    # Plot X and X_pred=F^-1(H)
    which_cmap = cm.coolwarm
    if 'solar' in self.path or 'traffic' in self.path:
        markersize = 20
        lwidth = 0.025
        X = plt_dict[0]
        vars = torch.var(X, dim=[0, 2]).cpu().detach()
        vars, idx = torch.sort(vars, descending=True)
        # All between 0 and 1
        vars = ((vars-vars.min())/(vars.max()-vars.min()))
        cutoff = 0.7
        vars[vars > cutoff] = vars[vars > cutoff]**2  # Make them lighter
        vars = torch.flip(vars, dims=(0,)).numpy()
    else:
        markersize = 2
        lwidth = 0.075
        vars = np.linspace(0, 1, V_tmp)
    print(f'1st Var to Last Var, lightest to darkest: {vars}')
    colors = np.tile(which_cmap(vars), (X.shape[0], 1))
    if V_tmp == 1:
        # Two-moon or 8_gaussian
        if '8_gaussian' in self.path:
            colors = np.repeat('r', N)
            colors[(Y == 1).cpu().detach().numpy().flatten()] = 'm'
            colors[(Y == 2).cpu().detach().numpy().flatten()] = 'y'
            colors[(Y == 3).cpu().detach().numpy().flatten()] = 'k'
        elif 'three_moon' in self.path:
            colors = np.repeat('black', N)
            colors[(Y == 1).cpu().detach().numpy().flatten()] = 'blue'
            colors[(Y == 2).cpu().detach().numpy().flatten()] = 'red'
        else:
            colors = np.repeat('black', N)
            colors[(Y == 1).cpu().detach().numpy().flatten()] = 'blue'
    for j in range(len(title_dict)):
        ax, ax1 = axs[j], axs[0]
        if j > 1:
            ax2 = axs[2]
        XorH = plt_dict[j]
        XorXpred_tmp = XorH.flatten(start_dim=0, end_dim=1).numpy()
        if self.C == 1:
            XorXpred_tmp = np.c_[XorXpred_tmp, np.zeros(XorXpred_tmp.shape)]
        if self.V > 1 or (self.V == 1 and self.C > 2):
            ax.plot(XorXpred_tmp[:, 0], XorXpred_tmp[:, 1],
                    linestyle='dashed', linewidth=lwidth)
        ax.scatter(XorXpred_tmp[:, 0],
                   XorXpred_tmp[:, 1], color=colors, s=markersize)
        ax.set_title(title_dict[j])
        # # Uncomment if we want the subplots to have fixed axes according to True X
        # if j < 2:
        #     X_tmp = plt_dict[0].flatten(start_dim=0, end_dim=1).numpy()
        #     ax.set_xlim(left=X_tmp[:, 0].min(), right=X_tmp[:, 0].max())
        #     ax.set_ylim(bottom=X_tmp[:, 1].min(), top=X_tmp[:, 1].max())
        if j == 1:
            ax.get_shared_x_axes().join(ax1, ax)
            ax.get_shared_y_axes().join(ax1, ax)
        if j == 3:
            ax.get_shared_x_axes().join(ax2, ax)
            ax.get_shared_y_axes().join(ax2, ax)
            ax.set_title(
                f'{title_dict[j]}, L_g is {L_g_now}')
    fig.tight_layout()
    self.fig_gen = fig
    plt.show()


def plt_generation_fig_competitor(self, X, X_pred, Y, H=None, H_pred=None):
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['figure.titlesize'] = 24
    X = X.cpu()
    N = X.shape[0]
    # num_to_plot = 1000 if '8_gaussian' in self.path else 600
    # if self.V > 1 or self.C > 2:
    #     num_to_plot = 100
    num_to_plot = N
    X, X_pred, Y = X[:num_to_plot], X_pred[:num_to_plot], Y[:num_to_plot]
    N = X.shape[0]
    if self.final_viz:
        plt_dict = {0: X_pred}
        title_dict = {0: r'$\hat{X}|Y=G^{-1}(H, Y)$'}
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        H, H_pred = H[:num_to_plot], H_pred[:num_to_plot]
        plt_dict = {0: X, 1: X_pred, 2: H, 3: H_pred}
        title_dict = {
            0: r'$X|Y$', 1: r'$\hat{X}|Y=G^{-1}(H, Y)$', 2: r'$H$', 3: r'$\hat{H}=G(X, Y)$'}
        fig, axs = plt.subplots(1, 4, figsize=(4 * 4, 4))
    V_tmp = X.shape[1]
    # Plot X and X_pred=F^-1(H)
    which_cmap = cm.coolwarm
    if 'solar' in self.path or 'traffic' in self.path:
        markersize = 20
        lwidth = 0.025
        vars = torch.var(X, dim=[0, 2]).cpu().detach()
        vars, idx = torch.sort(vars, descending=True)
        # All between 0 and 1
        vars = ((vars-vars.min())/(vars.max()-vars.min()))
        cutoff = 0.7
        vars[vars > cutoff] = vars[vars > cutoff]**2  # Make them lighter
        vars = torch.flip(vars, dims=(0,)).numpy()
    else:
        lwidth = 0.075
        vars = np.linspace(0, 1, V_tmp)
    print(f'1st Var to Last Var, lightest to darkest: {vars}')
    colors = np.tile(which_cmap(vars), (X.shape[0], 1))
    if V_tmp == 1:
        # Two-moon or 8_gaussian
        if '8_gaussian' in self.path:
            colors = np.repeat('r', N)
            colors[(Y == 1).cpu().detach().numpy().flatten()] = 'm'
            colors[(Y == 2).cpu().detach().numpy().flatten()] = 'y'
            colors[(Y == 3).cpu().detach().numpy().flatten()] = 'k'
        else:
            colors = np.repeat('black', N)
            colors[(Y == 1).cpu().detach().numpy().flatten()] = 'blue'
    for j in range(len(title_dict)):
        if len(title_dict) > 1:
            ax, ax1 = axs[j], axs[0]
        if j > 1:
            ax2 = axs[2]
        XorH = plt_dict[j]
        XorXpred_tmp = XorH.flatten(start_dim=0, end_dim=1).numpy()
        if self.C == 1:
            XorXpred_tmp = np.c_[XorXpred_tmp, np.zeros(XorXpred_tmp.shape)]
        if self.V > 1 or (self.V == 1 and self.C > 2):
            ax.plot(XorXpred_tmp[:, 0], XorXpred_tmp[:, 1],
                    linestyle='dashed', linewidth=lwidth)
        if 'solar' in self.path or 'traffic' in self.path:
            ax.scatter(XorXpred_tmp[:, 0],
                       XorXpred_tmp[:, 1], color=colors, s=markersize)
        else:
            ax.scatter(XorXpred_tmp[:, 0],
                       XorXpred_tmp[:, 1], color=colors)
        ax.set_title(title_dict[j])
        # # Uncomment if we want the subplots to have fixed axes according to True X
        # if j < 2:
        #     X_tmp = plt_dict[0].flatten(start_dim=0, end_dim=1).numpy()
        #     ax.set_xlim(left=X_tmp[:, 0].min(), right=X_tmp[:, 0].max())
        #     ax.set_ylim(bottom=X_tmp[:, 1].min(), top=X_tmp[:, 1].max())
        if j == 1:
            ax.get_shared_x_axes().join(ax1, ax)
            ax.get_shared_y_axes().join(ax1, ax)
        if j == 3:
            ax.get_shared_x_axes().join(ax2, ax)
            ax.get_shared_y_axes().join(ax2, ax)
    fig.tight_layout()
    self.fig_gen = fig
    plt.show()


def losses_and_error_plt_real_data_on_graph(self):
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['font.size'] = 18
    plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['legend.fontsize'] = 18
    # Quick plot
    if np.min(self.classify_error_ls_train) == 1:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(self.loss_g_ls_train,  label=r'Training', color='black')
        ax.plot(self.loss_g_ls_test, label=r'Test', color='blue')
        ax.set_title('Negative likelihood')
        ax.legend()
    else:
        fig, ax = plt.subplots(1, 3, figsize=(
            8, 4), constrained_layout=True)
        ax[0].plot(self.loss_g_ls_train,  label=r'Training', color='black')
        if min(self.loss_g_ls_test) > 0:
            ax[0].plot(self.loss_g_ls_test, label=r'Test', color='blue')
            ax[2].plot(self.loss_c_ls_test,  label=r'Test', color='blue')
            # ax[2].plot(self.classify_error_ls_test,  label=r'Test', color='blue')
            ax[1].plot(self.loss_w2_ls_test,
                       label=r'Training', color='black')
        ax[2].plot(self.loss_c_ls_train,
                   label=r'Training', color='black')
        # ax[2].plot(self.classify_error_ls_train,
        #            label=r'Training', color='black')
        ax[1].plot(self.loss_w2_ls_train,
                   label=r'Training', color='black')
        ax[0].set_title('Negative likelihood')
        ax[2].set_title(r'$\mu \cdot$Classification Loss')
        # ax[2].set_title('Classification Error')
        ax[1].set_title(r'$\gamma \cdot W_2$')
        for ax_now in ax:
            ax_now.legend()
    plt.show()
    plt.close()
    return fig


##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
