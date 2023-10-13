# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import cupy as cp
from pykdtree.kdtree import KDTree
import torch

import util_misc
import util_clustering


def kc_init_v3(n_kernel, points, points_weight, cmin, cmax, s_dims, device, points_sampling=1, reg_sampling=2, weight_exp=1, weight_thres=0, n_iter=300, **kwargs):
    '''Weighted Kmeans using util_clustering.KMeans'''
    p, d = points.shape

    # Subsample points
    if points_sampling > 1:
        points = points.view(*s_dims.tolist(), d)
        points_weight = points_weight.view(*s_dims.tolist(), 1)
        if s_dims.shape[0] == 2:
            points = points[::points_sampling, ::points_sampling]
            points_weight = points_weight[::points_sampling, ::points_sampling]
        elif s_dims.shape[0] == 3:
            points = points[::points_sampling, ::points_sampling, ::points_sampling]
            points_weight = points_weight[::points_sampling, ::points_sampling, ::points_sampling]
        else:
            raise NotImplementedError
        points = points.reshape(-1, d)
        points_weight = points_weight.reshape(-1, 1)

    points = points.view(-1, d)
    points_weight = points_weight.view(-1)
    points_weight = points_weight / points_weight.mean()
    
    if weight_thres != 0:
        points_weight[points_weight <= weight_thres] = 0
    if weight_exp != 1:
        points_weight = points_weight ** weight_exp

    kc = []

    # Regularly put some kernels
    if reg_sampling > 0:
        kc_i, _, _ = util_misc.get_grid_points(((s_dims/reg_sampling).ceil()+1).to(torch.long).tolist(), 
            align_corners=True, vmin=cmin.tolist(), vmax=cmax.tolist())
        kc_i = kc_i.flip(-1)  # x y ...
        kc.append(kc_i)
        n_kernel -= kc_i.shape[0]

    if n_kernel > 0:
        mask = points_weight > 0
        estimator = util_clustering.KMeans(n_clusters=n_kernel, max_iter=n_iter)
        estimator.fit(points[mask], sample_weight=points_weight[mask], backend=1, gpu=device)
        kc.append(torch.tensor(estimator.centers, device=points.device))

    kc = torch.cat(kc)
    return kc, {}


def kw_init_v3(kc, points, points_weight, s_dims, device, alpha, points_sampling=1, weight_exp=1, weight_thres=0, rbf_type='ivq_a', **kwargs):
    '''
    For each kernel, use weighted mean covariance with points in its cluster.
    kc: [k, d], torch tensor
    '''
    p, d = points.shape

    # Subsample points
    if points_sampling > 1:
        points = points.view(*s_dims.tolist(), d)
        points = points[::points_sampling, ::points_sampling]
        points = points.reshape(-1, d)
        points_weight = points_weight.view(*s_dims.tolist(), 1)
        points_weight = points_weight[::points_sampling, ::points_sampling]
        points_weight = points_weight.reshape(-1, 1)

    points = points.view(-1, d)
    points_weight = points_weight.view(-1)
    points_weight = points_weight / points_weight.mean()
    
    if weight_thres != 0:
        points_weight[points_weight <= weight_thres] = 0
    if weight_exp != 1:
        points_weight = points_weight ** weight_exp

    mask = points_weight > 0
    points = points[mask]
    points_weight = points_weight[mask]

    kc = kc.cpu().numpy()
    kdtree = KDTree(kc)
    _, labels = util_clustering.query_chunked(kdtree, points.cpu().numpy(), k=1, sqr_dists=True, 
        chunk_size=int(2e8), return_dist=False)  # [p]

    points_offset = points.cpu() - kc[labels]  # [p d]
    if rbf_type.endswith('_a') or rbf_type.endswith('_f'):
        points_data = points_offset[..., None] * points_offset[:, None, :]  # [p d d], covariance
    elif rbf_type.endswith('_d'):
        points_data = (points_offset**2)[..., None]  # [p d 1]
    elif rbf_type.endswith('_s'):
        points_data = (points_offset**2).sum(-1, keepdims=True)[..., None]  # [p 1 1]
    else:
        raise NotImplementedError
    del points_offset

    with cp.cuda.Device(device):
        labels = cp.asarray(labels)
        points_weight = cp.asarray(points_weight)
        kw_sq = torch.zeros([kc.shape[0], *points_data.shape[1:]], dtype=torch.float32, device=points.device)
        for i in range(points_data.shape[1]):
            for j in range(points_data.shape[2]):
                kw_sq_ij, count = util_clustering.reduce_within_clusters_chunked(cp.asarray(points_data[:, i, j]), 
                    kc.shape[0], labels, points_weight, chunk_size=int(1e8))
                kw_sq_ij /= alpha
                if rbf_type.endswith('_a') or rbf_type.endswith('_f'):
                    if i == j:
                        kw_sq_ij[count==0] = 1e-10
                    else:
                        kw_sq_ij[count==0] = 0
                else:
                    kw_sq_ij[count==0] = 1e-10
                kw_sq[:, i, j] = torch.tensor(kw_sq_ij, dtype=torch.float32, device=points.device)
                del kw_sq_ij, count

        del labels, points_weight
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    if rbf_type == 'nlin_s' or rbf_type == 'nlin_d':
        kw_sq[kw_sq < 0] = 1e-10

    return kw_sq
