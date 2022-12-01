import math
import numpy as np
import torch
import torch.nn as nn
import pdb


def resflow_logdet(self, x, edge_index=None, edge_weight=None):
    '''
        From Residual Flow
        Returns g(x) and logdet|d(x+g(x))/dx|
    '''
    x = x.requires_grad_(True)
    with torch.enable_grad():
        if edge_index is not None:
            g = self.bottleneck_block(x, edge_index, edge_weight)
        else:
            g = self.bottleneck_block(x)
        n_dist = 'poisson'
        if n_dist == 'geometric':
            geom_p = torch.sigmoid(0.5).item()
            def sample_fn(m): return geometric_sample(geom_p, m)
            def rcdf_fn(k, offset): return geometric_1mcdf(geom_p, k, offset)
        elif n_dist == 'poisson':
            lamb = 2.
            def sample_fn(m): return poisson_sample(lamb, m)
            def rcdf_fn(k, offset): return poisson_1mcdf(lamb, k, offset)

        # Unbiased estimation.
        n_samples = sample_fn(1)
        n_exact_terms = 2
        n_power_series = max(n_samples) + n_exact_terms
        def coeff_fn(k): return 1 / rcdf_fn(k, n_exact_terms) * \
            sum(n_samples >= k - n_exact_terms) / len(n_samples)

        ####################################
        # Power series with trace estimator.
        ####################################
        vareps = torch.randn_like(x)

        # Choose the type of estimator.
        estimator_fn = neumann_logdet_estimator

        # Do backprop-in-forward to save memory.
        g, logdetgrad = mem_eff_wrapper(
            estimator_fn, g, self.bottleneck_block, x, n_power_series, vareps, coeff_fn, True)
        return g, logdetgrad.view(-1, 1)


#####################
# Logdet Estimators
#####################

def mem_eff_wrapper(estimator_fn, g, gnet, x, n_power_series, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn, g, x, n_power_series, vareps, coeff_fn, training,
        * list(gnet.parameters())
    )


class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, g, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(
                g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, * \
                dg_params = torch.autograd.grad(
                    g, [x] + g_params, grad_g, allow_unused=True, retain_graph=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple(
                [g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(
                djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.contiguous().view(
        x.shape[0], -1) * vareps.contiguous().view(x.shape[0], -1), 1)
    return logdetgrad


# -------- Helper distribution functions --------
# These take python ints or floats, not PyTorch tensors.


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


# -------------- Helper functions --------------


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)

#####################
#####################
