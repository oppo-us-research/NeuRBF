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

import GPUtil
import numpy as np
import torch
import math
import trimesh
import mcubes
from einops import rearrange


comb2d = [[0,0],[1,0],[0,1],[1,1]]
comb3d = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]


def get_grid_points(dims, align_corners, vmin=-1, vmax=1, device='cpu'):
    num_dim = len(dims)
    if not (isinstance(vmin, tuple) or isinstance(vmin, list)):
        vmin = [vmin] * num_dim
    assert len(vmin) == num_dim
    if not (isinstance(vmax, tuple) or isinstance(vmax, list)):
        vmax = [vmax] * num_dim
    assert len(vmax) == num_dim

    if align_corners:
        side_interval = [(vmax[i] - vmin[i]) / (dim_i - 1) if dim_i > 1 
            else vmax[i] - vmin[i] for i, dim_i in enumerate(dims)]
        points = torch.meshgrid([torch.linspace(vmin[i], vmax[i], steps=dim_i, device=device) if dim_i > 1 
            else torch.tensor([(vmin[i] + vmax[i]) / 2], device=device) for i, dim_i in enumerate(dims)], indexing='ij')
    else:
        side_interval = [(vmax[i] - vmin[i]) / dim_i for i, dim_i in enumerate(dims)]
        points = torch.meshgrid([torch.linspace(vmin[i], vmax[i], 
            steps=dim_i + 1, device=device)[:-1] + side_interval[i] / 2
            for i, dim_i in enumerate(dims)], indexing='ij')
    points_grid = torch.stack(points, dim=-1)
    points = points_grid.reshape(-1, len(dims))
    side_interval = torch.tensor(side_interval, device=device)

    return points, points_grid, side_interval
    # points: (prod(dims), num_dim), order in the last dim: [dim0, dim1, ...].
    # points_grid: (*dims, num_dim), order in the last dim: [dim0, dim1, ...].
    # side_interval: (num_dim), order: [dim0, dim1, ...].


def get_lower_int_power(x, d):
    out = int(x**(1/d))
    if (out+1)**d == x:
        out+=1
    return out**d


def get_rbf_params_per_kernel(rbf_type, in_dim, kc_mult=2):
    if rbf_type.endswith('_a'):
        return int(in_dim*kc_mult + (in_dim+1)*in_dim/2)
    elif rbf_type.endswith('_f'):
        return int(in_dim*kc_mult + in_dim**2)
    elif rbf_type.endswith('_d'):
        return int(in_dim*kc_mult + in_dim)
    elif rbf_type.endswith('_s'):
        return int(in_dim*kc_mult + 1)
    else:
        raise NotImplementedError


def kw_to_kw_sq(kw, kernel_type):
    return kw**2


def kw_to_ks(kw, kernel_type, is_scalar=False):
    if kernel_type.startswith('nlin_') and \
        (is_scalar or (kernel_type[-2:] != '_a' and kernel_type[-2:] != '_f')):
        # Used to avoid power and then sqrt
        ks = 1 / kw
    else:
        kw_sq = kw_to_kw_sq(kw, kernel_type)
        ks = kw_sq_to_ks(kw_sq, kernel_type, is_scalar)
    return ks


def kw_sq_to_ks(kw_sq, kernel_type, is_scalar=False):
    if kernel_type[-2:] == '_a' and not is_scalar:
        if isinstance(kw_sq, np.ndarray):
            ks = np.linalg.inv(kw_sq)
        else:
            ks = torch.linalg.inv(kw_sq).contiguous()
    elif kernel_type[-2:] == '_f' and not is_scalar:
        d = kw_sq.shape[-1]
        if isinstance(kw_sq, np.ndarray):
            cov_inv = np.linalg.inv(kw_sq)  # [... d d]
            w, v = np.linalg.eig(cov_inv)  # [... d], [... d d]
            w = w[..., None]**0.5 * np.eye(d).reshape(*([1]*(len(w.shape)-1)), d, d)  # [... d d]
            ks = (v @ w).swapaxes(-1, -2)
        else:
            cov_inv = torch.linalg.inv(kw_sq).contiguous()  # [... d d]
            w, v = torch.linalg.eig(cov_inv)  # [... d], [... d d]
            w, v = w.real, v.real
            w = torch.diag_embed(w**0.5)  # [... d d]
            ks = (v @ w).swapaxes(-1, -2)  # [... d d]
    else:
        if kernel_type.startswith('nlin_'):
            ks = 1 / kw_sq**0.5
        else:
            ks = 1 / kw_sq
    return ks


def ks_to_kw_sq(ks, kernel_type, is_scalar=False):
    if kernel_type.endswith('_a') and not is_scalar:
        if isinstance(ks, np.ndarray):
            kw_sq = np.linalg.inv(ks)
        else:
            kw_sq = torch.linalg.inv(ks)
    elif kernel_type.endswith('_f') and not is_scalar:
        if isinstance(ks, np.ndarray):
            kw_sq = np.linalg.inv(ks.swapaxes(-1, -2) @ ks)
        else:
            kw_sq = torch.linalg.inv(ks.swapaxes(-1, -2) @ ks)
    else:
        if kernel_type.startswith('nlin_'):
            kw_sq = 1 / ks**2
        else:
            kw_sq = 1 / ks
    return kw_sq


def normalize_kw_sq(kw_sq, rbf_type):
    # Normalize kw so that rbf has value 0.5 when dist=kw
    if rbf_type.startswith('nlin_'):
        kw_sq = kw_sq * 4
    elif rbf_type.startswith('ivmq_'):
        kw_sq = kw_sq / 3
    return kw_sq


def clip_kw_sq(data, kernel_type, cmin, cmax, dims, is_ks, is_flat, vmin_factor=100, vmax_factor=10, vmin_min=1e-3):
    '''
    cmin: (n_dim), tensor, last dim order: x y ...
    cmax: (n_dim), tensor, last dim order: x y ...
    dims: (n_dim), tensor, last dim order: x y ...
    '''
    with torch.no_grad():
        vmin = ((cmax - cmin) / dims / 2 / vmin_factor).to(data.device)  # [d]
        vmax = ((cmax - cmin) / vmax_factor).to(data.device)  # [d]
        if vmin_min is not None:
            vmin.clip_(min=vmin_min)
        if not is_ks:  # data is kw_sq
            vmin, vmax = kw_to_kw_sq(vmin, kernel_type), kw_to_kw_sq(vmax, kernel_type)
        else:  # data is ks
            vmin, vmax = kw_to_ks(vmax, kernel_type, True), kw_to_ks(vmin, kernel_type, True)

        if kernel_type[-2:] == '_a' or (kernel_type[-2:] == '_f' and not is_ks):
            if not is_flat:
                # Diagonal elements
                for i in range(dims.shape[0]):
                    data[..., i, i].clip_(vmin[i], vmax[i])
                # Off-diagonal elements
                for i in range(dims.shape[0]):
                    for j in range(dims.shape[0]):
                        if i == j:
                            continue
                        val = (data[..., i, i] * data[..., j, j]).sqrt() * 0.999
                        data[..., i, j].clip_(min=-val, max=val)
            else:
                # Diagonal elements
                for i in range(dims.shape[0]):
                    data[..., i*dims.shape[0]+i].clip_(vmin[i], vmax[i])
                # Off-diagonal elements
                for i in range(dims.shape[0]):
                    for j in range(dims.shape[0]):
                        if i == j:
                            continue
                        val = (data[..., i*dims.shape[0]+i] * data[..., j*dims.shape[0]+j]).sqrt() * 0.999
                        data[..., i*dims.shape[0]+j].clip_(min=-val, max=val)
        elif kernel_type[-2:] == '_f' and is_ks:
            if not is_flat:
                for i in range(dims.shape[0]):
                    for j in range(dims.shape[0]):
                        data[..., i, j].clip_(-vmax[j], vmax[j])
            else:
                for i in range(dims.shape[0]):
                    for j in range(dims.shape[0]):
                        data[..., i*dims.shape[0]+j].clip_(-vmax[j], vmax[j])
        elif kernel_type[-2:] == '_d':
            data.clip_(vmin[None, :, None], vmax[None, :, None])
        elif kernel_type[-2:] == '_s':
            data.clip_(vmin.max(), vmax.min())
        else:
            raise NotImplementedError


def rescale_kc(kc, block_centers, patch_size, input_size):
    return kc * (np.array(patch_size) - 1)[None, None, :] / np.array(input_size[:-1])[None, None, :] + block_centers


def rescale_kw_sq(kw_sq, patch_size, input_size):
    if kw_sq.ndim < 4:
        if kw_sq.ndim == 2:
            kw_sq = np.tile(kw_sq[..., None], (1, 1, len(patch_size)))
        return kw_sq * ((np.array(patch_size) - 1)[None, None, :] / np.array(input_size[:-1])[None, None, :]) ** 2
    else:
        raise NotImplementedError


def get_multi_index_nearest(coords, dims, interval, vmin, vmax, align_corners=True):
    """
    coords: (..., n_dim), last dim order: x y ...
    dims: (n_dim), tensor, last dim order: x y ...
    interval: (n_dim), tensor, last dim order: x y ...
    vmin: (n_dim), tensor, last dim order: x y ...
    vmax: (n_dim), tensor, last dim order: x y ...
    return: (..., n_dim), last dim order: x y ...
    """
    n_dim = coords.shape[-1]
    interval = interval.expand([*([1]*(coords.dim()-1)), -1])  # (..., n_dim)
    if align_corners:
        sub = (coords - vmin.expand([*([1]*(coords.dim()-1)), -1]) + interval/2) / interval  # (..., n_dim)
    else:
        sub = (coords - vmin.expand([*([1]*(coords.dim()-1)), -1])) / interval  # (..., n_dim)
    mask = (sub - sub.round()).abs() < 1e-3
    sub[mask] = sub[mask].round()
    sub[~mask] = sub[~mask].floor()
    sub = sub.clip(min=0).clip(max=dims.expand([*([1]*(sub.dim()-1)), n_dim])-1)  # (..., n_dim)
    return sub


def get_multi_index_multi_nearest(coords, dims, interval, vmin, vmax, align_corners=True):
    """
    coords: (..., n_dim), last dim order: x y ...
    dims: (n_dim), tensor, last dim order: x y ...
    interval: (n_dim), tensor, last dim order: x y ...
    vmin: (n_dim), tensor, last dim order: x y ...
    vmax: (n_dim), tensor, last dim order: x y ...
    return: (..., n_dim), last dim order: x y ...
    """
    n_dim = coords.shape[-1]
    interval = interval.expand([*([1]*(coords.dim()-1)), -1])  # (..., n_dim)
    if align_corners:
        sub = (coords - vmin.expand([*([1]*(coords.dim()-1)), -1]) + interval/2) / interval  # (..., n_dim)
    else:
        sub = (coords - vmin.expand([*([1]*(coords.dim()-1)), -1])) / interval  # (..., n_dim)
    mask = (sub - sub.round()).abs() < 1e-3
    sub[mask] = sub[mask].round()
    sub[~mask] = sub[~mask].floor()
    sub = sub.clip(min=0).clip(max=dims.expand([*([1]*(sub.dim()-1)), n_dim])-1)  # (..., n_dim)
    sub = sub[..., None, :]

    # coords on block edges are assigned to all adjacent blocks
    for i in range(n_dim):
        temp = coords[..., i] / interval[..., i]
        temp = (temp - temp.round()).abs()
        mask = (temp < 1e-3) * (coords[..., i] > vmin[i]) * (coords[..., i] < vmax[i])
        sub_i = torch.ones_like(sub) * dims.max()
        sub_i[mask] = sub[mask]
        sub_i[mask, ..., i] -= 1
        sub = torch.cat([sub, sub_i], dim=-2)

    return sub


def get_multi_index_nearest_lower(coords, dims, interval, vmin, vmax, align_corners=True):
    """
    Get the multi index of nearest lower neighbor.
    coords: (..., n_dim), last dim order: x y ...
    dims: (n_dim), tensor, last dim order: x y ...
    interval: (n_dim), tensor, last dim order: x y ...
    vmin: (n_dim), tensor, last dim order: x y ...
    vmax: (n_dim), tensor, last dim order: x y ...
    return: (..., n_dim), last dim order: x y ...
    """
    n_dim = coords.shape[-1]
    interval = interval.expand([*([1]*(coords.dim()-1)), -1])  # (..., n_dim)
    vmin = vmin.expand([*([1]*(coords.dim()-1)), -1])  # (..., n_dim)
    dims = dims.expand([*([1]*(coords.dim()-1)), n_dim])  # (..., n_dim)

    if align_corners:
        sub = torch.div(coords - vmin, interval, rounding_mode='floor')  # (..., n_dim)
    else:
        sub = torch.div(coords - vmin - interval/2, interval, rounding_mode='floor')  # (..., n_dim)
    sub = sub.clip(min=0).clip(max=dims-2)  # (..., n_dim)
    return sub.to(torch.long)


def get_multi_index_first_ring(coords, dims, interval, vmin, vmax, align_corners=True):
    """
    coords: (..., n_dim), last dim order: x y ...
    dims: (n_dim), tensor, last dim order: x y ...
    interval: (n_dim), last dim order: x y ...
    vmin: (n_dim), tensor, last dim order: x y ...
    vmax: (n_dim), tensor, last dim order: x y ...
    return: (..., n_dim), last dim order: x y ...
    """
    sub = get_multi_index_nearest_lower(coords, dims, interval, vmin, vmax, align_corners=True)

    n_dim = coords.shape[-1]
    if n_dim == 2:
        comb = comb2d
    elif n_dim == 3:
        comb = comb3d
    else:
        raise NotImplementedError
    sub = sub[..., None, :] + \
        torch.tensor(comb, device=sub.device, dtype=torch.long).expand([*([1]*(sub.dim()-1)), -1, n_dim])  # (..., n_ngb, n_dim)
    return sub


def ravel_multi_index(sub, dims):
    """
    sub: (..., n_dim), tensor, last dim order: x y ...
    dims: (n_dim), tensor, last dim order: x y ...
    """
    idx = (sub * (dims.cumprod(-1) / dims).to(torch.long).expand(*([1]*(sub.dim()-1)), -1)).sum(-1)
    return idx  # (...)


def ravel_multi_index_loop(sub, dims):
    """
    sub: (..., n_dim), tensor, last dim order: x y ...
    dims: (n_dim), tensor, last dim order: x y ...
    """
    mult_factor = (dims.cumprod(-1) / dims).to(torch.long)
    idx = sub[..., 0] * mult_factor[0]
    for i in range(1, len(dims)):
        idx += sub[..., i] * mult_factor[i]
    idx = idx.to(torch.long)
    return idx  # (...)


def select_devices(devices, force_reselect=True, excludeID=[1]):
    if isinstance(devices, str):
        if not devices.endswith('#'):
            if not force_reselect:
                return devices
            else:
                n_device = len(devices.split(','))
        else:
            n_device = int(devices[:-1])
    elif isinstance(devices, list) or isinstance(devices, tuple):
        if not force_reselect:
            return devices
        else:
            n_device = len(devices)

    devices = GPUtil.getAvailable(order='memory', limit=n_device, maxLoad=1.0, maxMemory=0.75, 
        includeNan=False, excludeID=excludeID, excludeUUID=[])  # Find available GPUs by sorting memory usage
    if len(devices) < n_device:
        raise ValueError('Not enough available GPUs.')
    return devices


def count_parameters(model):
    if isinstance(model, torch.nn.Module) or isinstance(model, torch.nn.ModuleList):
        out = np.array([sum(p.numel() for _, p in model.named_parameters() if p.requires_grad), 
            sum(p.numel() for _, p in model.named_parameters() if not p.requires_grad), 0])
    elif isinstance(model, torch.nn.Parameter):
        out = np.array([model.numel() if model.requires_grad else 0, 
            model.numel() if not model.requires_grad else 0, 0])
    else:
        raise NotImplementedError
    out[-1] = out[0] + out[1]
    return out


def extract_geometry(sdf, thres, cmin, cmax):
    if torch.is_tensor(sdf):
        sdf = sdf.squeeze().cpu().numpy()
    assert(len(sdf.shape)==3)
    assert(sdf.shape[0]==sdf.shape[1] and sdf.shape[0]==sdf.shape[2])
    sdf = -sdf.transpose((2, 1, 0))  # w h d, - for inside, + for outside
    resolution = sdf.shape[0]
    vertices, faces = mcubes.marching_cubes(sdf, thres)
    vertices = vertices / (resolution - 1.0) * (cmax - cmin)[None, :] + cmin[None, :]
    return vertices, faces


def extract_mesh(sdf, thres, cmin, cmax):
    vertices, faces = extract_geometry(sdf, thres, cmin, cmax)
    mesh = trimesh.Trimesh(vertices, faces, process=False)  # important, process=True leads to seg fault...
    return mesh


def random_shuffle_chunked(n, chunk_size=10000*10000, device=0):
    device_cuda = 0 if device == 'cpu' else device
    if n <= chunk_size:
        out = torch.randperm(n, device=device_cuda)
    else:
        n_chunk = math.ceil(n / chunk_size)
        out = torch.multinomial(torch.ones([chunk_size, n_chunk], device=device_cuda), n_chunk) * chunk_size
        out += torch.arange(chunk_size, device=device_cuda)[:, None].expand(-1, n_chunk)
        for i in range(n_chunk):
            out[:, i] = out[torch.randperm(chunk_size, device=device_cuda), i]
        out = out.reshape(-1)
        out = out[out < n]
    return out.to(device_cuda)


def stratified_shuffle_chunked(s_dims, b_dims, chunk_size=10000*5000, device=0):
    '''
    s_dims: tensor, ... h w
    b_dims: tensor, ... h w
    '''
    device_cuda = 0 if device == 'cpu' else device
    d = s_dims.shape[0]
    s_dims_new = ((s_dims / b_dims).ceil() * b_dims).to(torch.long)

    _, points_grid, _ = get_grid_points(s_dims_new.tolist(), align_corners=True, vmin=0, vmax=(s_dims_new-1).tolist(), 
        device=device)  # ... h w d
    points_grid = rearrange(points_grid, '(b0 p0) (b1 p1) d -> (b0 b1) (p0 p1) d', b0=b_dims[0], b1=b_dims[1])
    b, p = points_grid.shape[:-1]

    b_chunk = math.floor(chunk_size / p)
    if b_chunk >= b:
        order = torch.multinomial(torch.ones([b, p], device=device_cuda), p).to(device)  # b p
    else:
        order = torch.zeros([b, p], dtype=torch.long, device=device)
        prob = torch.ones([b_chunk, p], device=device_cuda)
        for i in range(0, b, b_chunk):
            b_chunk_i = min(b_chunk, b - i)
            if b_chunk_i != b_chunk:
                prob = torch.ones([b_chunk_i, p], device=device_cuda)
            order[i:i+b_chunk_i, :] = torch.multinomial(prob, p).to(order.device)

    points_grid = points_grid.gather(1, order[..., None].expand(-1, -1, d)).to(torch.long)  # b p d
    s_dims = s_dims.to(device)
    mask = (points_grid < s_dims[None, None]).all(dim=-1)  # b p
    points_grid = ravel_multi_index_loop(points_grid.flip(-1), s_dims.flip(-1))  # b p
    return points_grid.to(dtype=torch.int, device=device_cuda), mask.to(device_cuda)
