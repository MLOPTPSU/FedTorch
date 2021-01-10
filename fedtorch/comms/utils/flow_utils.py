# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

import torch


SCALE_QUANTIZE, ZERO_POINT_QUANTIZE, DTYPE_QUANTIZE = 0.001, 0, torch.qint8
def get_current_epoch(args):
    if args.growing_batch_size:
        args.epoch_ = args.local_data_seen / args.num_samples_per_epoch
    else:
        args.epoch_ = args.local_index / args.num_batches_train_per_device_per_epoch
    args.epoch = int(args.epoch_)


def get_current_local_step(args):
    """design a specific local step adjustment schme based on lr_decay_by_epoch
    """
    try:
        return args.local_steps[args.epoch]
    except:
        return args.local_steps[-1]


def is_stop(args):
    if args.stop_criteria == 'epoch':
        return args.epoch >= args.num_epochs
    elif args.stop_criteria == 'iteration':
        return args.local_index >= args.num_iterations_per_worker


def is_sync_fed(args):
    if args.federated_sync_type == 'local_step':
        local_step = get_current_local_step(args)
        return args.local_index % local_step == 0
    elif args.federated_sync_type == 'epoch':
        return args.epoch_ % args.num_epochs_per_comm == 0
    else:
        raise NotImplementedError

def is_sync_fed_robust(args):
    local_step = get_current_local_step(args)
    return args.local_index % local_step**2 == 0

def update_client_epoch(args):
    args.client_epoch_total += args.local_index / args.num_batches_train_per_device_per_epoch
    return



def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v)[0],dims=(0,))
    cssv = torch.cumsum(u,dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho=0.0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


def projection_simplex_pivot(v, z=1, random_state=None):
    rs = np.random.RandomState(random_state)
    n_features = len(v)
    U = np.arange(n_features)
    s = 0
    rho = 0
    while len(U) > 0:
        G = []
        L = []
        k = U[rs.randint(0, len(U))]
        ds = v[k]
        for j in U:
            if v[j] >= v[k]:
                if j != k:
                    ds += v[j]
                    G.append(j)
            elif v[j] < v[k]:
                L.append(j)
        drho = len(G) + 1
        if s + ds - (rho + drho) * v[k] < z:
            s += ds
            rho += drho
            U = L
        else:
            U = G
    theta = (s - z) / float(rho)
    return np.maximum(v - theta, 0)

def projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
    lower = 0
    upper = np.max(v)
    current = np.inf

    for it in range(max_iter):
        if np.abs(current) / z < tau and current < 0:
            break

        theta = (upper + lower) / 2.0
        w = np.maximum(v - theta, 0)
        current = np.sum(w) - z
        if current <= 0:
            upper = theta
        else:
            lower = theta
    return w

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    
    return w

def zero_copy(model,rnn=False):
    if rnn:
        model.hidden = None
    tmp_model = deepcopy(model)
    for tp in tmp_model.parameters():
        tp.data = torch.zeros_like(tp.data)
    if rnn:
        model.init_hidden()
    return tmp_model

def quantize_tensor(x, num_bits=8, adaptive=False, info=None):
    qmin = -2.**(num_bits-1)
    qmax =  2.**(num_bits-1) - 1.
    if adaptive:
        min_val, max_val, mean_val = x.min(), x.max(), x.mean()

        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0.0:
            scale=0.001

        initial_zero_point = qmin - (min_val - mean_val) / scale

        zero_point = 0
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point
        zero_point = int(zero_point)
    else:
        if info is not None:
            scale=info[0]
            zero_point=info[1]
            mean_val=info[2]
        else:
            scale=SCALE_QUANTIZE
            zero_point=ZERO_POINT_QUANTIZE
            mean_val=0.0
    
    q_x = zero_point + (x - mean_val) / scale
    q_x.clamp_(qmin, qmax).round_()
    if num_bits == 8:
        q_x = q_x.round().char()
    elif num_bits == 16:
        q_x = q_x.round().short()
    return q_x, torch.tensor([scale, zero_point, mean_val])


def dequantize_tensor(q_x, info=None):
    if info is None:
        return SCALE_QUANTIZE * (q_x.float() - ZERO_POINT_QUANTIZE)
    else:
        return info[0] * (q_x.float() - info[1]) + info[2]

def size_tensor(x):
    return x.element_size() * x.nelement() / 1e6


def compress_tensor(x, r=0.5, comp_type='topk'):
    s = torch.tensor(x.shape).type(torch.int16)
    x_f = x.flatten()
    k = int(len(x_f)*r/2)
    if k == 0:
        raise ValueError("Compression ratio is too low!")
    if comp_type=='topk':
        v,i = x_f.abs().topk(k)
    elif comp_type=='random':
        i = torch.randperm(len(x_f))[:k]
    v = x_f[i]
    i = i.type(torch.int32)
    return v,i,s

def decompress_tensor(v,i,s):
    s = s.tolist()
    x_d = torch.zeros(np.prod(s)).to(v.device)
    x_d[i.long()] = v
    x_d = x_d.reshape(s)
    return x_d


def alpha_update(model_local, model_personal,alpha, eta):
    grad_alpha = 0
    for l_params, p_params in zip(model_local.parameters(), model_personal.parameters()):
        dif = p_params.data - l_params.data
        grad = alpha * p_params.grad.data + (1-alpha)*l_params.grad.data
        grad_alpha += dif.view(-1).T.dot(grad.view(-1))
    
    grad_alpha += 0.02 * alpha
    alpha_n = alpha - eta*grad_alpha
    alpha_n = np.clip(alpha_n.item(),0.0,1.0)
    return alpha_n