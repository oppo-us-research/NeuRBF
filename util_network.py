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

import torch
from functools import partial
import time
import trimesh
import kornia

import util_misc
import util_init


class PE(torch.nn.Module):
    """
    positional encoding
    """
    def __init__(self, P):
        """
        P: (d, F) encoding matrix
        """
        super().__init__()
        self.register_buffer("P", P)

    @property
    def out_dim(self):
        return self.P.shape[1]*2

    def forward(self, x):
        """
        x: (p, d)
        """
        x_ = (x[..., None] * self.P[None]).sum(-2) # (p, F)
        return torch.cat([torch.sin(x_), torch.cos(x_)], -1) # (p, 2*F)


@torch.jit.script
def gaussian_activation(x, a):
    return torch.exp(-x**2/(2*a**2))


@torch.jit.script
def scaledsin_activation(x, a):
    return torch.sin(x*a)


@torch.jit.script
def rbf_nlin_s_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, 1)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc
    out = (1 - offset.abs() * ks).clamp(0, 1).prod(-1)
    return out, offset


@torch.jit.script
def rbf_nlin_d_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # [..., k, d]
    out = (1 - offset.abs() * ks).clamp(0, 1).prod(-1)
    return out, offset


@torch.jit.script
def rbf_nlin_f_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # [..., k, d]
    out = (1 - (offset[..., None, :] * ks).sum(-1).abs()).clamp(0, 1).prod(-1)
    return out, offset


@torch.jit.script
def rbf_ivq_s_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, 1)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc
    out = (1 / (1 + offset.pow(2).sum(-1) * ks[..., 0]))
    return out, offset


@torch.jit.script
def rbf_ivq_d_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = (1 / (1 + (offset.pow(2) * ks).sum(-1)))
    return out, offset


@torch.jit.script
def rbf_ivq_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = 1 / (1 + ((offset[..., None] * ks).sum(-2) * offset).sum(-1))
    return out, offset


def rbf_ivmq_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = 1 / torch.sqrt(1 + ((offset[..., None] * ks).sum(-2) * offset).sum(-1))
    return out, offset


def rbf_ivd_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = 1 / (torch.sqrt(((offset[..., None] * ks).sum(-2) * offset).sum(-1)) + 1e-10)
    return out, offset


def rbf_ivc_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = 1 / (((offset[..., None] * ks).sum(-2) * offset).sum(-1).sqrt()**3 + 1e-30)
    return out, offset


def rbf_gauss_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    d = x.shape[-1]
    offset = x[..., None, :] - kc  # (..., k, d)
    out = torch.exp(-0.5*((offset[..., None] * ks).sum(-2) * offset).sum(-1))
    return out, offset


def rbf_sgauss2_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    d = x.shape[-1]
    offset = x[..., None, :] - kc  # (..., k, d)
    out = torch.exp(-(0.5*((offset[..., None] * ks).sum(-2) * offset).sum(-1))**2)
    return out, offset


def rbf_nsgauss2_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    d = x.shape[-1]
    offset = x[..., None, :] - kc  # (..., k, d)
    out = torch.exp(-0.5*((offset[..., None] * ks).sum(-2) * offset).sum(-1))**2
    return out, offset


def rbf_markov_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    d = x.shape[-1]
    offset = x[..., None, :] - kc  # (..., k, d)
    out = torch.exp(-((offset[..., None] * ks).sum(-2) * offset).sum(-1).sqrt())
    return out, offset


def rbf_expsin_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = torch.exp(-torch.sin(((offset[..., None] * ks).sum(-2) * offset).sum(-1).sqrt()))
    return out, offset


def rbf_qd_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = 1 + ((offset[..., None] * ks).sum(-2) * offset).sum(-1)
    return out, offset


def rbf_mqd_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = torch.sqrt(1 + ((offset[..., None] * ks).sum(-2) * offset).sum(-1))
    return out, offset


def rbf_phs1_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = ((offset[..., None] * ks).sum(-2) * offset).sum(-1).sqrt()
    return out, offset


def rbf_phs1_s_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, 1)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = torch.sqrt((offset**2).sum(-1))
    return out, offset


def rbf_phs3_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = ((offset[..., None] * ks).sum(-2) * offset).sum(-1).sqrt()**3
    return out, offset


def rbf_vep_a_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = 1 - ((offset[..., None] * ks).sum(-2) * offset).sum(-1)
    return out, offset


def rbf_cos_s_fb(x, kc, ks):
    """
        Inputs:
            x: (..., d)
            kc: (..., k, d)
            ks: (..., k, d, d)
        Outputs:
            (..., k)
    """
    offset = x[..., None, :] - kc  # (..., k, d)
    out = torch.nn.functional.cosine_similarity(x[..., None, :], kc, dim=-1)
    return out, offset


def fix_params(model, fix_params):
    for name, p in model.named_parameters():
        if name.split('.')[0] in fix_params:
            p.requires_grad = False


def get_param_groups(model):
    param_groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if name.startswith('backbone'):
                k = 'dec'
            elif name.startswith('encoder'):
                k = 'hg'
            elif name.startswith('hg0'):
                k = 'hg0'
            elif name.startswith('lc0'):
                k = 'lc0'
            elif name.startswith('lcb0'):
                k = 'lcb0'
            elif name.startswith('kc0'):
                k = 'kc0'
            elif name.startswith('ks0'):
                k = 'ks0'
            else:
                raise NotImplementedError
            if k not in param_groups:
                param_groups[k] = []
            param_groups[k].append(p)
            print(k, name, p.shape)
    return param_groups


def parse_optim_type(optim_type):
    if optim_type == 'SparseAdam':
        return torch.optim.SparseAdam
    elif optim_type == 'Adam':
        return torch.optim.Adam
    elif optim_type == 'RAdam':
        return torch.optim.RAdam
    elif optim_type == 'SGD':
        return torch.optim.SGD
    else:
        raise NotImplementedError


def configure_optimizers(param_groups, hparams):
    optims = {}
    for k in param_groups.keys():
        hparams_k = hparams[k]
        optim_type = hparams_k['type']
        optim_fn = parse_optim_type(optim_type)
        if optim_type == 'SGD':
            optims[k] = optim_fn(param_groups[k], lr=hparams_k['lr'], weight_decay=hparams_k['wd'])
        elif optim_type == 'SparseAdam':
            optims[k] = optim_fn(param_groups[k], lr=hparams_k['lr'], betas=hparams_k['betas'], eps=hparams_k['eps'])
        else:
            optims[k] = optim_fn(param_groups[k], lr=hparams_k['lr'], betas=hparams_k['betas'], eps=hparams_k['eps'], weight_decay=hparams_k['wd'])
    return optims


def parse_lr_sch_type(lr_sch_type):
    if lr_sch_type == 'ngp':
        return lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.33**max(0, step//10000-1))
    elif lr_sch_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR
    else:
        raise NotImplementedError


def configure_lr_schedulers(optims, hparams):
    lr_schs = {}
    for k in optims.keys():
        hparams_k = hparams[k]
        lr_sch_type = hparams_k['type']
        lr_sch_fn = parse_lr_sch_type(lr_sch_type)
        if lr_sch_type == 'ngp':
            lr_schs[k] = lr_sch_fn(optims[k])
        else:
            lr_schs[k] = lr_sch_fn(optims[k], T_max=hparams_k['T_max'], eta_min=optims[k].param_groups[0]['lr']/hparams_k['gamma'])
    return lr_schs


def init_kc_i(n_kernel, rbf_type, dataset, kc_init_config, device):
    with torch.no_grad():
        init_type = kc_init_config['type']
        kc_init_fn = None if init_type == 'none' else \
            partial(eval('util_init.kc_init_' + init_type), **kc_init_config)

        kc = None
        out_other = {}
        if kc_init_fn is not None:
            points = dataset.points.cpu().reshape(-1, dataset.points.shape[-1])
            if kc_init_config['weight_src'] == 'gt_grad':
                points_weight = dataset.gt_grad.cpu()
            elif kc_init_config['weight_src'] == 'gt':
                points_weight = dataset.gt.cpu().reshape(-1, dataset.gt.shape[-1])
                points_weight = (points_weight.abs() <= kc_init_config['weight_src_thres']).to(torch.float)
                points_weight[kc_init_config['weight_src_npts']:] = 0
            elif kc_init_config['weight_src'] == 'gt_inv':
                points_weight = dataset.gt.cpu().reshape(-1, dataset.gt.shape[-1])
                points_weight = 1 / (points_weight.abs() + kc_init_config['weight_src_thres'])
                points_weight[kc_init_config['weight_src_npts']:] = 0
            else:
                raise NotImplementedError
            kc, out_other = kc_init_fn(n_kernel, points=points, 
                points_weight=points_weight, cmin=dataset.cmin, cmax=dataset.cmax, 
                s_dims=dataset.s_dims, device=device, rbf_type=rbf_type)

    return kc, out_other


def init_ks_i(rbf_type, dataset, kw_init_config, kc, out_other, device):
    with torch.no_grad():
        init_type = kw_init_config['type']
        kw_init_fn = None if init_type == 'none' else \
            partial(eval('util_init.kw_init_' + init_type), **kw_init_config)

        ks = None
        if kw_init_fn is not None:
            points = dataset.points.cpu().reshape(-1, dataset.points.shape[-1])
            if kw_init_config['weight_src'] == 'gt_grad':
                points_weight = dataset.gt_grad.cpu()
            elif kw_init_config['weight_src'] == 'gt':
                points_weight = dataset.gt.cpu().reshape(-1, dataset.gt.shape[-1])
                points_weight = (points_weight.abs() <= kw_init_config['weight_src_thres']).to(torch.float)
                points_weight[kw_init_config['weight_src_npts']:] = 0
            elif kw_init_config['weight_src'] == 'gt_inv':
                points_weight = dataset.gt.cpu().reshape(-1, dataset.gt.shape[-1])
                points_weight = 1 / (points_weight.abs() + kw_init_config['weight_src_thres'])
                points_weight[kw_init_config['weight_src_npts']:] = 0
            else:
                raise NotImplementedError
            kw_sq = kw_init_fn(kc, in_other=out_other, points=points, 
                points_weight=points_weight, cmin=dataset.cmin, cmax=dataset.cmax, 
                s_dims=dataset.s_dims, device=device, rbf_type=rbf_type)
            kw_sq = util_misc.normalize_kw_sq(kw_sq, rbf_type)
            util_misc.clip_kw_sq(kw_sq, rbf_type, dataset.cmin.flip(-1), dataset.cmax.flip(-1), 
                dataset.s_dims.flip(-1), is_ks=False, is_flat=False, vmin_min=None)
            ks = util_misc.kw_sq_to_ks(kw_sq, rbf_type)

    return ks


def init_rbf_params(model, dataset, kc_init_config, kw_init_config, device):
    with torch.no_grad():
        for k in model.rbf_suffixes:
            n_kernel = getattr(model, f'kc{k}').weight.shape[0]

            print(f'Initializing kc{k}...')
            t = time.time()
            kc, out_other = init_kc_i(n_kernel, model.rbf_type, dataset, kc_init_config[k], device)
            if kc is not None:
                getattr(model, f'kc{k}').weight.data = kc
                print(f'Init kc{k}: {time.time() - t}')
            else:
                kc = getattr(model, f'kc{k}').weight.data

            print(f'Initializing ks{k}...')
            t = time.time()
            ks = init_ks_i(model.rbf_type, dataset, kw_init_config[k], kc, out_other, device)
            if ks is not None:
                getattr(model, f'ks{k}').weight.data = ks.reshape(n_kernel, -1)
                print(f'Init ks{k}: {time.time() - t}')


def init_nerf_kc_i(n_kernel, rbf_type, points, density, features, cmin, cmax, s_dims, kc_init_config, device):
    with torch.no_grad():
        init_type = kc_init_config['type']
        kc_init_fn = None if init_type == 'none' else \
            partial(eval('util_init.kc_init_' + init_type), **kc_init_config)

        kc = None
        out_other = {}
        if kc_init_fn is not None:
            points = points.cpu().reshape(-1, points.shape[-1])

            alphas = 1 - torch.exp(-density)

            features = (features - features.min()) / (features.max() - features.min())
            features = features.reshape(*s_dims.tolist(), -1)
            features_grad = kornia.filters.SpatialGradient3d(mode='diff', order=1)(features.movedim(-1, 0)[None])[0]  # [c 3 d h w]
            features_grad = features_grad.movedim((0, 1), (-2, -1)).pow(2).sum(dim=[-2, -1]).sqrt()[..., None]  # [d h w 1]

            points_weight = alphas * features_grad.reshape(-1, 1)

            kc, out_other = kc_init_fn(n_kernel, points=points, points_weight=points_weight, 
                cmin=cmin, cmax=cmax, s_dims=s_dims, device=device, rbf_type=rbf_type)

    return kc, out_other


def init_nerf_ks_i(rbf_type, points, density, features, cmin, cmax, s_dims, kw_init_config, kc, out_other, device):
    with torch.no_grad():
        init_type = kw_init_config['type']
        kw_init_fn = None if init_type == 'none' else \
            partial(eval('util_init.kw_init_' + init_type), **kw_init_config)

        ks = None
        if kw_init_fn is not None:
            points = points.cpu().reshape(-1, points.shape[-1])

            alphas = 1 - torch.exp(-density)

            features = (features - features.min()) / (features.max() - features.min())
            features = features.reshape(*s_dims.tolist(), -1)
            features_grad = kornia.filters.SpatialGradient3d(mode='diff', order=1)(features.movedim(-1, 0)[None])[0]  # [c 3 d h w]
            features_grad = features_grad.movedim((0, 1), (-2, -1)).pow(2).sum(dim=[-2, -1]).sqrt()[..., None]  # [d h w 1]

            points_weight = alphas * features_grad.reshape(-1, 1)

            kw_sq = kw_init_fn(kc, in_other=out_other, points=points, points_weight=points_weight, 
                cmin=cmin, cmax=cmax, s_dims=s_dims, device=device, rbf_type=rbf_type)
            kw_sq = util_misc.normalize_kw_sq(kw_sq, rbf_type)
            util_misc.clip_kw_sq(kw_sq, rbf_type, cmin.flip(-1), cmax.flip(-1), s_dims.flip(-1), 
                is_ks=False, is_flat=False, vmin_factor=100, vmax_factor=10)
            ks = util_misc.kw_sq_to_ks(kw_sq, rbf_type)

    return ks


def init_nerf_rbf_params(model, points, density, features, kc_init_config, kw_init_config, device):
    with torch.no_grad():
        for k in model.rbf_suffixes:
            n_kernel = getattr(model, f'kc{k}').weight.shape[0]

            print(f'Initializing kc{k}...')
            t = time.time()
            kc, out_other = init_nerf_kc_i(n_kernel, model.rbf_type, points, density, features, model.cmin, model.cmax, 
                model.s_dims, kc_init_config[k], device)
            if kc is not None:
                getattr(model, f'kc{k}').weight.data = kc
                print(f'Init kc{k}: {time.time() - t}')
            else:
                kc = getattr(model, f'kc{k}').weight.data

            print(f'Initializing ks{k}...')
            t = time.time()
            ks = init_nerf_ks_i(model.rbf_type, points, density, features, model.cmin, model.cmax, model.s_dims, 
                kw_init_config[k], kc, out_other, device)
            if ks is not None:
                getattr(model, f'ks{k}').weight.data = ks.reshape(n_kernel, -1)
                print(f'Init ks{k}: {time.time() - t}')
