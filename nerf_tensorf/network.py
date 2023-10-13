import os
import time
import math
import numpy as np
import torch.nn as nn
import mcubes
import trimesh
from pykdtree.kdtree import KDTree
import kornia

from thirdparty.tensorf.models.tensorBase import *
import thirdparty.tensorf.utils as utils
import util_misc
import util_network
import util_clustering


class TensorVMSplitRBF(TensorBase):
    """
    Modified from https://github.com/apchenstu/TensoRF/blob/4ec894dc1341a2201fe13ae428631b58458f105d/models/tensoRF.py#L139
    """
    def __init__(self, aabb, gridSize, device, **kargs):
        self.n_level = kargs['args'].n_level
        self.level_types = list(kargs['args'].level_types)
        if len(self.level_types) == 1:
            self.level_types *= self.n_level
        assert(len(self.level_types) == self.n_level)
        for x in self.level_types:
            assert(x in ['vm', 'g'])
        
        self.N_voxel_min = kargs['args'].resol_min ** 3

        self.density_n_comp = list(kargs['density_n_comp'])
        self.app_n_comp = list(kargs['appearance_n_comp'])
        if len(self.density_n_comp) == 1:
            self.density_n_comp *= self.n_level
        if len(self.app_n_comp) == 1:
            self.app_n_comp *= self.n_level
        assert(len(self.density_n_comp) == self.n_level)
        assert(len(self.app_n_comp) == self.n_level)
        self.density_n_comp_total = sum(self.density_n_comp) * 3
        self.app_n_comp_total = sum(self.app_n_comp) * 3

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]

        args = kargs['args']
        rbf_config = args.rbf_config
        self.lc0_dim = rbf_config['lc0_dim']
        self.lcd0_dim = rbf_config['lcd0_dim']
        self.get_init_data = rbf_config['get_init_data']
        self.n_kernel = rbf_config['n_kernel']
        self.rbf_type = rbf_config['rbf_type']
        self.ref_config = rbf_config['ref_config']
        self.in_dim = 3
        self.kc_mult = 1  # kc_mult=2 means extra copy of initial kc

        gridSize_ref = [round(self.ref_config['N_voxel_final']**(1/3)) - 1] * 3  # Not actual value, just for init rbf params
        gridSize_ref = np.array(gridSize_ref).astype(np.int64)
        gridSize_final = [round(args.N_voxel_real_final**(1/3))] * 3
        gridSize_final = np.array(gridSize_final).astype(np.int64)
        if self.n_level > 1:
            down_factor = np.exp((np.log(np.prod(gridSize_final)) - np.log(self.N_voxel_min)) / 3 / (self.n_level - 1))
        else:
            down_factor = 1
        gridSizes = []
        for i in range(self.n_level):
            gridSize_i = np.round(gridSize_final / down_factor**i).astype(np.int64)
            gridSizes.append(gridSize_i)

        n_params_ref = self.count_tensorf_params(self.ref_config['n_lamb_sigma'], [gridSize_ref]) + \
            self.count_tensorf_params(self.ref_config['n_lamb_sh'], [gridSize_ref])
        n_params_tensorf = self.count_tensorf_params(self.density_n_comp, gridSizes) + \
            self.count_tensorf_params(self.app_n_comp, gridSizes)
        if self.n_kernel == 'auto':
            n_params_rbf = n_params_ref - n_params_tensorf
            self.n_kernel = int(n_params_rbf // (self.lc0_dim + self.lcd0_dim + util_misc.get_rbf_params_per_kernel(
                self.rbf_type, self.in_dim, self.kc_mult)))
        print('n_kernel:', self.n_kernel)

        self.rbf_train_start = False
        if self.n_kernel > 0 and not self.get_init_data:
            self.rbf_train_start = True

        super(TensorVMSplitRBF, self).__init__(aabb, gridSize, device, **kargs)

        self.args = args
        self.cmin = torch.tensor([-1, -1, -1])  # ... y x, aabb
        self.cmax = torch.tensor([1, 1, 1])  # ... y x, aabb
        self.register_buffer('cmin_gpu', self.cmin.clone())  # ... y x, aabb
        self.register_buffer('cmax_gpu', self.cmax.clone())  # ... y x, aabb
        self.init_steps = rbf_config['init_steps']
        self.init_data_fp = rbf_config['init_data_fp']
        self.init_rbf = rbf_config['init_rbf']
        self.s_dims = rbf_config['s_dims']
        if self.s_dims != 'aabb':
            self.s_dims = torch.tensor(self.s_dims)  # ... h w, used for initializing rbf
            self.rbf_normalize_pts = True
        else:
            self.rbf_normalize_pts = False
        self.rbf_lc0_normalize = rbf_config['rbf_lc0_normalize']
        self.point_nn_kernel = rbf_config['point_nn_kernel']
        ks_alpha = rbf_config['ks_alpha']
        self.lc_init = rbf_config['lc_init']
        self.lcd_init = rbf_config['lcd_init']
        self.lcb_init = rbf_config['lcb_init']
        self.rbf_suffixes = rbf_config['rbf_suffixes']
        self.pe_lc0_freq = rbf_config['pe_lc0_freq']
        self.pe_lc0_rbf_freq = rbf_config['pe_lc0_rbf_freq']
        self.pe_lc0_rbf_keep = rbf_config['pe_lc0_rbf_keep']
        if self.pe_lc0_rbf_keep == 'all': self.pe_lc0_rbf_keep = self.lc0_dim

        if len(self.pe_lc0_freq) >= 2:
            self.pe_lc0_freqs = torch.linspace(np.log2(self.pe_lc0_freq[0]), np.log2(self.pe_lc0_freq[1]), 
                                              self.app_dim, device=0)
            self.pe_lc0_freqs = torch.exp2(self.pe_lc0_freqs)
        else:
            self.pe_lc0_freqs = None

        if len(self.pe_lc0_rbf_freq) >= 2 and self.pe_lc0_rbf_keep < self.lc0_dim:
            self.pe_lc0_rbf_freqs = torch.linspace(np.log2(self.pe_lc0_rbf_freq[0]), np.log2(self.pe_lc0_rbf_freq[1]), 
                                                   self.lc0_dim - self.pe_lc0_rbf_keep, device=0)
            self.pe_lc0_rbf_freqs = torch.exp2(self.pe_lc0_rbf_freqs)
        else:
            self.pe_lc0_rbf_freqs = None

        fix_params = rbf_config['fix_params']
        self.kc_init_config = rbf_config['kc_init_config']
        self.kw_init_config = rbf_config['kw_init_config']
        self.kc_init_regular = {}
        for k, v in self.kc_init_config.items():
            if v['type'] == 'none':
                self.kc_init_regular[k] = True
            else:
                self.kc_init_regular[k] = False
        sparse_embd_grad = False

        # Init rbf
        self.kc0 = nn.ModuleList()
        self.ks0 = nn.ModuleList()
        self.lc0 = nn.ModuleList()
        if self.rbf_train_start:
            if self.kc_init_regular['0']:
                self.n_kernel = util_misc.get_lower_int_power(self.n_kernel, self.in_dim)
            self.rbf_fn = eval(f'util_network.rbf_{self.rbf_type}_fb')
            if self.lc0_dim > 0:
                self.lc_dims = [[self.n_kernel, self.lc0_dim]]
                self.kc0, self.ks0, self.lc0, self.ks_dims, k_dims, kc_interval = self.create_rbf_params(
                    self.rbf_type, self.n_kernel, self.in_dim, self.lc0_dim, sparse_embd_grad, self.cmin, self.cmax, ks_alpha, is_bag=False)
                nn.init.uniform_(self.lc0.weight, self.lc_init[0], self.lc_init[1])
                self.register_buffer(f'k_dims_0', k_dims)
                self.register_buffer(f'kci0', kc_interval)
                if self.init_rbf:
                    self.init_rbf_params(kargs['init_data'])
            if self.lcd0_dim > 0:
                self.lcd_dims = [[self.n_kernel, self.lcd0_dim]]
                _, _, self.lcd0, _, _, _ = self.create_rbf_params(
                    self.rbf_type, self.n_kernel, self.in_dim, self.lcd0_dim, sparse_embd_grad, self.cmin, self.cmax, ks_alpha, is_bag=False)
                nn.init.uniform_(self.lcd0.weight, self.lcd_init[0], self.lcd_init[1])
        
        if self.lcb_init is not None:
            self.lcb0 = torch.nn.Parameter(torch.zeros((1, self.app_n_comp_total)))
            nn.init.uniform_(self.lcb0.data, self.lcb_init[0], self.lcb_init[1])

        # Init decoder network
        self.init_render_func(self.shadingMode, self.app_dim, self.pos_pe, self.view_pe, self.fea_pe, 
                              self.featureC, self.device, args=args)

        # Fix params
        util_network.fix_params(self, fix_params)


    def count_tensorf_params(self, n_component, gridSizes):
        n_params = 0
        for j in range(len(gridSizes)):
            gridSize = gridSizes[j]
            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]
                n_params += n_component[j] * gridSize[mat_id_1] * gridSize[mat_id_0]
                n_params += n_component[j] * gridSize[vec_id]
        return n_params


    def init_svd_volume(self, res, device):
        gridSize = np.array(res)

        if self.n_level > 1:
            down_factor = np.exp((np.log(np.prod(gridSize)) - np.log(self.N_voxel_min)) / 3 / (self.n_level - 1))
        else:
            down_factor = 1
        self.gridSizes = []
        self.density_plane = torch.nn.ParameterList()
        self.density_line = torch.nn.ParameterList()
        self.app_plane = torch.nn.ParameterList()
        self.app_line = torch.nn.ParameterList()
        for i in range(self.n_level):
            gridSize_i = np.round(gridSize / down_factor**i).astype(np.int64)
            self.gridSizes.append(gridSize_i)
            level_type = self.level_types[i]
            density_plane, density_line = self.init_one_svd(
                self.density_n_comp[i], gridSize_i, 0.1, device, level_type)
            app_plane, app_line = self.init_one_svd(
                self.app_n_comp[i], gridSize_i, 0.1, device, level_type)
            self.density_plane += density_plane
            self.density_line += density_line
            self.app_plane += app_plane
            self.app_line += app_line

        if self.rbf_train_start and self.lcd0_dim > 0:
            self.basis_mat_ds = torch.nn.Linear(2, 1, bias=False).to(device)
        
        if self.rbf_train_start:
            self.app_n_comp_total += self.lc0_dim
        if self.app_dim is not None:
            self.basis_mat = torch.nn.Linear(self.app_n_comp_total, self.app_dim, bias=False).to(device)
        else:
            self.app_dim = self.app_n_comp_total


    def init_one_svd(self, n_component, gridSize, scale, device, level_type):
        plane_coef, line_coef = [], []
        if level_type == 'vm':
            for i in range(len(self.vecMode)):
                vec_id = self.vecMode[i]
                mat_id_0, mat_id_1 = self.matMode[i]
                plane_coef.append(torch.nn.Parameter(
                    scale * torch.randn((1, n_component, gridSize[mat_id_1], gridSize[mat_id_0]))))
                line_coef.append(
                    torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        else:
            plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component * 3, gridSize[2], gridSize[1], gridSize[0]))))
            plane_coef += [torch.nn.Parameter(torch.zeros([0])), torch.nn.Parameter(torch.zeros([0]))]
            line_coef += [torch.nn.Parameter(torch.zeros([0])), torch.nn.Parameter(torch.zeros([0])), 
                          torch.nn.Parameter(torch.zeros([0]))]

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        params_vm = []
        params_g = []
        for i in range(self.n_level):
            level_type = self.level_types[i]
            if level_type == 'vm':
                for j in range(3):
                    params_vm.append(self.density_line[i * 3 + j])
                    params_vm.append(self.density_plane[i * 3 + j])
                    params_vm.append(self.app_line[i * 3 + j])
                    params_vm.append(self.app_plane[i * 3 + j])
            elif level_type == 'g':
                params_g.append(self.density_plane[i * 3])
                params_g.append(self.app_plane[i * 3])

        grad_vars = []
        if len(params_vm) > 0:
            grad_vars += [{'params': params_vm, 'lr': lr_init_spatialxyz}]
        if len(params_g) > 0:
            grad_vars += [{'params': params_g, 'lr': lr_init_spatialxyz * self.args.lr_g_factor}]
        if hasattr(self, 'lc0'):
            grad_vars.append({'params': self.lc0.parameters(), 'lr': self.args.rbf_config['lr_config']['lc0']})
        if hasattr(self, 'lcd0'):
            grad_vars.append({'params': self.lcd0.parameters(), 'lr': self.args.rbf_config['lr_config']['lcd0']})
        if hasattr(self, 'lcb0'):
            grad_vars.append({'params': [self.lcb0], 'lr': self.args.rbf_config['lr_config']['lcb0']})
        if hasattr(self, 'basis_mat'):
            grad_vars += [{'params': self.basis_mat.parameters(), 'lr': self.args.lr_basis_mat}]
        if hasattr(self, 'basis_mat_ds'):
            grad_vars.append({'params': self.basis_mat_ds.parameters(), 'lr': lr_init_network})

        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            if self.density_plane[idx].shape[0] != 0:
                total = total + torch.mean(torch.abs(self.density_plane[idx]))
            if self.density_line[idx].shape[0] != 0:
                total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            if self.density_plane[idx].shape[0] != 0:
                total = total + reg(self.density_plane[idx]) * 1e-2
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            if self.app_plane[idx].shape[0] != 0:
                total = total + reg(self.app_plane[idx]) * 1e-2
        return total


    def compute_densityfeature(self, xyz_sampled, others={}):
        # xyz_sampled: [n_pts, 3], normalized to range [-1, 1] based on self.aabb

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)  # [3, n_pts, 1, 2]
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))  # [3, n_pts]
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)  # [3, n_pts, 1, 2]

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)  # [n_pts]
        for idx_plane in range(len(self.density_plane)):
            level_type = self.level_types[idx_plane // 3]
            if level_type == 'vm':
                idx_coord = idx_plane % 3
                # density_plane: [1, c, h, w]
                plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_coord]],
                                                    align_corners=True).view(-1, xyz_sampled.shape[0])  # [c, n_pts]
                # density_line: [1, c, d, 1]
                line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_coord]],
                                                align_corners=True).view(-1, xyz_sampled.shape[0])  # [c, n_pts]
                sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
            else:
                if self.density_plane[idx_plane].shape[0] != 0:
                    out = F.grid_sample(self.density_plane[idx_plane], xyz_sampled[None, :, None, None],
                                        align_corners=True).view(-1, xyz_sampled.shape[0])  # [c, n_pts]
                    sigma_feature = sigma_feature + torch.sum(out, dim=0)
        
        if self.rbf_train_start and self.lcd0_dim > 0:
            kernel_idx, rbf_out = others['kernel_idx'], others['rbf_out']
            out = self.lcd0(kernel_idx)  # [p k_topk d_lcd0]
            out = (out * rbf_out).sum(1)  # [p d_lcd0]
            sigma_feature = torch.stack([sigma_feature, out.sum(-1)], -1)
            sigma_feature = self.basis_mat_ds(sigma_feature)[..., 0]

        return sigma_feature


    def compute_appfeature(self, xyz_sampled, others={}):
        # xyz_sampled: [n_pts, 3], normalized to range [-1, 1] based on self.aabb

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        grid_coef_point = []
        for idx_plane in range(len(self.app_plane)):
            level_type = self.level_types[idx_plane // 3]
            if level_type == 'vm':
                idx_coord = idx_plane % 3
                plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_coord]],
                                                    align_corners=True).view(-1, *xyz_sampled.shape[:1]))  # [c, n_pts]
                line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_coord]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))  # [c, n_pts]
            else:
                if self.app_plane[idx_plane].shape[0] != 0:
                    grid_coef_point.append(F.grid_sample(self.app_plane[idx_plane], xyz_sampled[None, :, None, None],
                                                        align_corners=True).view(-1, xyz_sampled.shape[0]).T)  # [n_pts, c]
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        app_feature = (plane_coef_point * line_coef_point).T  # [n_pts, c]
        if len(grid_coef_point) > 0:
            app_feature = torch.cat([app_feature] + grid_coef_point, -1)

        # Forward rbf lc0
        if self.rbf_train_start:
            kernel_idx, rbf_out = others['kernel_idx'], others['rbf_out']
            if self.pe_lc0_rbf_freqs is not None:
                if self.pe_lc0_rbf_keep > 0:
                    rbf_out = torch.cat([rbf_out.expand(-1, -1, self.pe_lc0_rbf_keep), 
                        torch.sin(rbf_out * self.pe_lc0_rbf_freqs[None, None])], -1)  # [p k_topk d_lc0]
                else:
                    rbf_out = torch.sin(rbf_out * self.pe_lc0_rbf_freqs[None, None])  # [p k_topk d_lc0]
            out = self.lc0(kernel_idx)  # [p k_topk d_lc0]
            out = (out * rbf_out).sum(1)  # [p d_lc0]
            app_feature = torch.cat([app_feature, out], -1)

        if hasattr(self, 'lcb0'):
            app_feature = app_feature + self.lcb0

        if hasattr(self, 'basis_mat'):
            app_feature = self.basis_mat(app_feature)  # [n_pts, app_dim]

        if self.pe_lc0_freqs is not None:
            app_feature = app_feature + torch.sin(app_feature * self.pe_lc0_freqs[None])

        return app_feature


    def create_rbf_params(self, rbf_type, n_kernel, in_dim, lc_dim, sparse_embd_grad, cmin, cmax, ks_alpha, scale_grad_by_freq=False, is_bag=True):
        if rbf_type.endswith('_a') or rbf_type.endswith('_f'):
            ks_dims = [in_dim, in_dim]
        elif rbf_type.endswith('_d'):
            ks_dims = [in_dim]
        elif rbf_type.endswith('_s'):
            ks_dims = [1]
        else:
            raise NotImplementedError
        kc = torch.nn.Embedding(n_kernel, in_dim, scale_grad_by_freq=False, sparse=sparse_embd_grad)
        ks = torch.nn.Embedding(n_kernel, np.prod(ks_dims), scale_grad_by_freq=False, 
            sparse=sparse_embd_grad)
        if rbf_type.endswith('_a') or rbf_type.endswith('_f'):
            ks.weight.data = torch.eye(in_dim)[None, ...].repeat(n_kernel, 1, 1).reshape(n_kernel, -1)
        elif rbf_type.endswith('_d'):
            ks.weight.data[:] = 1
        elif rbf_type.endswith('_s'):
            ks.weight.data[:] = 1
        else:
            raise NotImplementedError

        if is_bag:
            lc = torch.nn.EmbeddingBag(n_kernel, lc_dim, scale_grad_by_freq=scale_grad_by_freq, 
                                       sparse=sparse_embd_grad, mode='sum')
        else:
            lc = torch.nn.Embedding(n_kernel, lc_dim, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse_embd_grad)

        # Initialize
        n_kernel_per_dim = int(math.ceil(n_kernel**(1/in_dim)))
        k_dims = [n_kernel_per_dim] * in_dim
        points, _, side_interval = util_misc.get_grid_points(k_dims, align_corners=True, 
            vmin=cmin.tolist(), vmax=cmax.tolist())
        points = points.flip(-1)  # x y ...
        side_interval = side_interval.flip(-1)  # x y ...
        kc.weight.data = points[:n_kernel]
        if rbf_type.startswith('nlin_'):
            if rbf_type.endswith('_s'):
                ks.weight.data *= 1 / side_interval.mean()
            elif rbf_type.endswith('_d'):
                ks.weight.data *= 1 / side_interval[None]
            elif rbf_type.endswith('_f'):
                ks.weight.data *= (1 / side_interval)[:, None].repeat(1, side_interval.shape[0]).view(1, -1)
            else:
                raise NotImplementedError
        else:
            if rbf_type.endswith('_s'):
                ks.weight.data *= 1 / (side_interval.mean() / 2) ** 2
            elif rbf_type.endswith('_d'):
                ks.weight.data *= 1 / (side_interval[None] / 2) ** 2
            elif rbf_type.endswith('_a'):
                ks.weight.data *= (1 / (side_interval / 2) ** 2)[:, None].repeat(1, side_interval.shape[0]).view(1, -1)
            else:
                raise NotImplementedError
        ks.weight.data *= ks_alpha

        return kc, ks, lc, ks_dims, torch.tensor(k_dims).flip(-1), side_interval


    def get_kc(self, suffix):
        return getattr(self, 'kc' + suffix).weight


    def get_ks(self, suffix):
        return getattr(self, 'ks' + suffix).weight.view(-1, *self.ks_dims)


    def clip_rbf_params(self):
        for k in self.rbf_suffixes:
            if hasattr(self, f'ks{k}.weight') and getattr(self, 'ks'+k).weight.requires_grad:
                if k == '0':
                    vmin_factor = 100
                    vmax_factor = 10
                else:
                    vmin_factor = 1
                    vmax_factor = 2
                util_misc.clip_kw_sq(
                    getattr(self, 'ks'+k).weight, self.rbf_type, self.cmin.flip(-1), self.cmax.flip(-1), 
                    self.s_dims.flip(-1), is_ks=True, is_flat=True, vmin_factor=vmin_factor, vmax_factor=vmax_factor)
                

    def init_rbf_params(self, init_data=None):
        # Get grid points within aabb
        if self.s_dims == 'aabb':
            if self.get_init_data:
                self.aabb_rbf = self.aabb
                aabb_size = self.aabb_rbf[1] - self.aabb_rbf[0]  # w h ...
                sl = aabb_size.prod() ** (1/3) / 512
                self.s_dims = (aabb_size / sl).round().long().flip(-1)  # ... h w
            else:
                points_weight = init_data['points_weight']
                self.aabb_rbf = init_data['aabb'].to(self.device)
                self.s_dims = init_data['s_dims']
                del init_data

            self.cmin = self.aabb_rbf[0].flip(-1).cpu()  # ... y x
            self.cmax = self.aabb_rbf[1].flip(-1).cpu()  # ... y x
            self.register_buffer('cmin_gpu', self.cmin.clone())  # ... y x
            self.register_buffer('cmax_gpu', self.cmax.clone())  # ... y x

            t = time.time()
            pts, _, _ = util_misc.get_grid_points(self.s_dims.tolist(), align_corners=True, 
                vmin=self.cmin.tolist(), vmax=self.cmax.tolist(), device=0)
            pts = pts.flip(-1)  # x y z
            print('Generate grid points:', time.time() - t)

            # Get density and features at these points
            if self.get_init_data:
                t = time.time()
                density_all, features_all = self.get_pts_data(pts, normalize_pts=True)
                if features_all.shape[-1] > 27:
                    shuffle = torch.randperm(features_all.shape[-1]).long()[:27]
                    features_all = features_all[:, shuffle]
                print('Get density and features at grid points:', time.time() - t)
                density, features = density_all.cpu().view(-1, 1), features_all.cpu()
                del density_all, features_all

                # Compute point weight
                density = 1 - torch.exp(-density)
                features = (features - features.min()) / (features.max() - features.min())
                features = features.reshape(*self.s_dims.tolist(), -1)
                features = kornia.filters.SpatialGradient3d(mode='diff', order=1)(features.movedim(-1, 0)[None])[0]  # [c 3 d h w]
                features = features.movedim((0, 1), (-2, -1)).pow(2).sum(dim=[-2, -1]).sqrt()[..., None]  # [d h w 1]
                points_weight = density * features.reshape(-1, 1)
                del density, features

                init_data = {'points_weight': points_weight, 
                             'aabb': self.aabb_rbf.cpu(), 's_dims': self.s_dims.cpu()}

                # # Save init data
                # os.makedirs(os.path.dirname(self.init_data_fp), exist_ok=True)
                # torch.save(init_data, self.init_data_fp)
                # density_all = density_all.reshape(*self.s_dims.tolist(), -1).permute(2, 1, 0, 3)
                # vertices, triangles = mcubes.marching_cubes(density_all.numpy()[..., 0], 1.)
                # vertices = vertices / (np.array(density_all.shape[:3]) - 1.0) * \
                #     aabb_size.cpu().numpy()[None] + self.aabb_rbf[0].cpu().numpy()[None]
                # trimesh.Trimesh(vertices, triangles, process=False).export(self.init_data_fp.split('.')[0] + '.ply')

                torch.cuda.empty_cache()
                return init_data
        else:
            raise NotImplementedError
            t = time.time()
            pts, _, _ = util_misc.get_grid_points(self.s_dims.tolist(), align_corners=True, 
                vmin=self.cmin.tolist(), vmax=self.cmax.tolist(), device=0)
            pts = pts.flip(-1)  # x y z
            print('Generate grid points:', time.time() - t)

            # Get density and features at these points
            if self.get_init_data:
                t = time.time()
                density_all, features_all = self.get_pts_data(pts)
                if features_all.shape[-1] > 27:
                    shuffle = torch.randperm(features_all.shape[-1]).long()[:27]
                    features_all = features_all[:, shuffle]
                print('Get density and features at grid points:', time.time() - t)
                density_all, features_all = density_all.cpu(), features_all.cpu()
                init_data = {'density': density_all, 'features': features_all, 
                            'aabb': self.aabb.cpu()}

                # # Save init data
                # os.makedirs(os.path.dirname(self.init_data_fp), exist_ok=True)
                # torch.save(init_data, self.init_data_fp)
                # density_all = density_all.reshape(*self.s_dims.tolist(), -1).permute(2, 1, 0, 3)
                # vertices, triangles = mcubes.marching_cubes(density_all.numpy()[..., 0], 1.)
                # vertices = vertices / (density_all.shape[0] - 1.0) * 2 - 1
                # trimesh.Trimesh(vertices, triangles, process=False).export(self.init_data_fp.split('.')[0] + '.ply')

                torch.cuda.empty_cache()
                return init_data
        
            density_all = init_data['density'].view(-1, 1)
            features_all = init_data['features']
            self.aabb_rbf = init_data['aabb'].to(self.device)
            del init_data

        # Init rbf parameters
        util_network.init_nerf_rbf_params(self, pts, points_weight, None, self.kc_init_config, 
                                          self.kw_init_config, device=0)
        del points_weight
        torch.cuda.empty_cache()

        # Build kd tree
        self.kdtree0 = KDTree(getattr(self, 'kc0').weight.detach().cpu().numpy())


    def step_after_iter(self, step):
        if step == self.init_steps and self.get_init_data:
            return self.init_rbf_params()


    def compute_rbf(self, xyz_sampled):
        out = {}

        # Forward rbf lc0
        if self.rbf_train_start:
            # Un-normalize
            pts = (xyz_sampled + 1) / 2 * (self.aabb[1] - self.aabb[0]) + self.aabb[0]
            # Re-normalize based on self.aabb_rbf
            if self.rbf_normalize_pts:
                pts = (pts - self.aabb_rbf[0]) * 2.0 / (self.aabb_rbf[1] - self.aabb_rbf[0]) - 1

            suffix = '0'
            if self.point_nn_kernel <= 0:  # Use all kernels for each point
                raise NotImplementedError
            else:
                kernel_idx = self.forward_kernel_idx(pts, None, suffix)
                rbf_out = self.forward_rbf(pts, kernel_idx, suffix)  # [p k_topk]
                if self.rbf_lc0_normalize:
                    rbf_out = rbf_out / (rbf_out.detach().sum(-1, keepdim=True) + 1e-8)
                rbf_out = rbf_out[..., None]  # [p k_topk 1]

            out['kernel_idx'] = kernel_idx
            out['rbf_out'] = rbf_out

        return out
    

    def forward_kernel_idx(self, x_g, point_idx, suffix):
        if hasattr(self, 'point_kernel_idx_' + suffix):
            # Use pre-computed knn
            point_kernel_idx = getattr(self, 'point_kernel_idx_' + suffix)
            kernel_idx = point_kernel_idx[point_idx.to(point_kernel_idx.device)].to(x_g.device)  # [p topk]
        elif self.kc_init_regular[suffix]:
            # Find first ring neighbors in regular grid
            kernel_idx = util_misc.get_multi_index_first_ring(x_g, 
                getattr(self, f'k_dims_{suffix}'), getattr(self, f'kci{suffix}'), 
                self.cmin_gpu.flip(-1), self.cmax_gpu.flip(-1), True)  # (p, n_ngb, n_in)
            kernel_idx = util_misc.ravel_multi_index(kernel_idx, getattr(self, f'k_dims_{suffix}'))  # (p, n_ngb)
        else:
            # Knn on the fly
            if not hasattr(self, 'kdtree' + suffix):
                setattr(self, 'kdtree' + suffix, KDTree(getattr(self, 'kc' + suffix).weight.detach().cpu().numpy()))
            _, kernel_idx = util_clustering.query_chunked(getattr(self, 'kdtree' + suffix), x_g.cpu().numpy(), 
                k=self.point_nn_kernel, sqr_dists=True, chunk_size=int(2e8), return_dist=False)
            kernel_idx = torch.tensor(kernel_idx.astype(np.int32), device=x_g.device)  # [p topk]
            if kernel_idx.ndim == 1:
                kernel_idx = kernel_idx[:, None]
            
        return kernel_idx


    def forward_rbf(self, x_g, kernel_idx, suffix):
        if kernel_idx is None:  # Use all kernels for each point
            raise NotImplementedError
        else:
            kc = getattr(self, 'kc' + suffix)(kernel_idx)  # [p k d]
            ks = getattr(self, 'ks' + suffix)(kernel_idx).view(*kernel_idx.shape, *self.ks_dims)  # [p k d d] or [p k 1]
            rbf_out, _ = self.rbf_fn(x_g, kc, ks)  # [p k_topk]
        return rbf_out


    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        res_target = np.array(res_target)
        if self.n_level > 1:
            down_factor = np.exp((np.log(np.prod(res_target)) - np.log(self.N_voxel_min)) / 3 / (self.n_level - 1))
        else:
            down_factor = 1

        for idx_level in range(self.n_level):
            level_type = self.level_types[idx_level]
            res_target_i = np.round(res_target / down_factor**idx_level).astype(np.int64)
            self.gridSizes[idx_level] = res_target_i
            res_target_i = res_target_i.tolist()
            if level_type == 'vm':
                for i in range(len(self.vecMode)):
                    vec_id = self.vecMode[i]
                    mat_id_0, mat_id_1 = self.matMode[i]
                    idx = idx_level * 3 + i
                    plane_coef[idx] = torch.nn.Parameter(
                        F.interpolate(plane_coef[idx].data, size=(res_target_i[mat_id_1], res_target_i[mat_id_0]), mode='bilinear',
                                    align_corners=True))
                    line_coef[idx] = torch.nn.Parameter(
                        F.interpolate(line_coef[idx].data, size=(res_target_i[vec_id], 1), mode='bilinear', align_corners=True))
            else:
                idx = idx_level * 3
                plane_coef[idx] = torch.nn.Parameter(
                    F.interpolate(plane_coef[idx].data, size=(res_target_i[2], res_target_i[1], res_target_i[0]), mode='trilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, res_target_real):
        if not self.args.no_upsample:
            self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target_real)
            self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target_real)

        self.update_stepSize(res_target)

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        aabb_size = self.aabb[1] - self.aabb[0]
        xyz_min, xyz_max = new_aabb
        gridSize = torch.from_numpy(self.gridSizes[0]).to(aabb_size.device)
        units = aabb_size / (gridSize - 1)

        t_l, b_r = (xyz_min - self.aabb[0]) / units, (xyz_max - self.aabb[0]) / units
        t_l, b_r = torch.floor(t_l).long(), torch.ceil(b_r).long() + 1
        b_r = torch.stack([b_r, gridSize]).amin(0)

        t_l_np = t_l.cpu().numpy()
        b_r_np = b_r.cpu().numpy()
        for idx_level in range(self.n_level):
            level_type = self.level_types[idx_level]
            if idx_level == 0:
                t_l_i = t_l_np
                b_r_i = b_r_np
            else:
                factor = self.gridSizes[idx_level] / self.gridSizes[0]
                t_l_i = np.floor(t_l_np * factor).astype(np.int64)
                b_r_i = np.ceil(b_r_np * factor).astype(np.int64)

            if level_type == 'vm':
                for i in range(len(self.vecMode)):
                    idx = idx_level * 3 + i
                    mode0 = self.vecMode[i]
                    self.density_line[idx] = torch.nn.Parameter(
                        self.density_line[idx].data[...,t_l_i[mode0]:b_r_i[mode0],:]
                    )
                    self.app_line[idx] = torch.nn.Parameter(
                        self.app_line[idx].data[...,t_l_i[mode0]:b_r_i[mode0],:]
                    )
                    mode0, mode1 = self.matMode[i]
                    self.density_plane[idx] = torch.nn.Parameter(
                        self.density_plane[idx].data[...,t_l_i[mode1]:b_r_i[mode1],t_l_i[mode0]:b_r_i[mode0]]
                    )
                    self.app_plane[idx] = torch.nn.Parameter(
                        self.app_plane[idx].data[...,t_l_i[mode1]:b_r_i[mode1],t_l_i[mode0]:b_r_i[mode0]]
                    )
            else:
                idx = idx_level * 3
                self.density_plane[idx] = torch.nn.Parameter(
                    self.density_plane[idx].data[...,t_l_i[2]:b_r_i[2],t_l_i[1]:b_r_i[1],t_l_i[0]:b_r_i[0]]
                    )
                self.app_plane[idx] = torch.nn.Parameter(
                    self.app_plane[idx].data[...,t_l_i[2]:b_r_i[2],t_l_i[1]:b_r_i[1],t_l_i[0]:b_r_i[0]]
                    )

        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.floor(t_l).long(), torch.ceil(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb

        voxels_orig = np.prod(self.gridSizes[0])
        target_res = utils.N_to_reso(voxels_orig, new_aabb)
        if self.args.scale_reso:
            target_res = utils.scale_reso(target_res, voxels_orig)
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, target_res)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, target_res)

        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
