# Modified from https://github.com/sarafridov/K-Planes/blob/main/plenoxels/models/kplane_field.py

import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from thirdparty.kplanes.ops.interpolation import grid_sample_wrapper
from thirdparty.kplanes.raymarching.spatial_distortions import SpatialDistortion

import mcubes
import trimesh
from pykdtree.kdtree import KDTree
import util_misc
import util_network
import util_init
import util_clustering


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class KPlaneRBFField(nn.Module):
    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        use_appearance_embedding: bool,
        appearance_embedding_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        density_activation: Callable,
        linear_decoder: bool,
        linear_decoder_layers: Optional[int],
        num_images: Optional[int],
        rbf_config: Dict,
        init_data: Optional[Dict],
    ) -> None:
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder

        self.in_dim = 3
        self.cmin = torch.tensor([-1, -1, -1])  # ... y x, aabb
        self.cmax = torch.tensor([1, 1, 1])  # ... y x, aabb
        self.register_buffer('cmin_gpu', self.cmin.clone())  # ... y x, aabb
        self.register_buffer('cmax_gpu', self.cmax.clone())  # ... y x, aabb
        self.get_init_data = rbf_config['get_init_data']
        self.init_steps = rbf_config['init_steps']
        self.init_data_fp = rbf_config['init_data_fp']
        self.init_rbf = rbf_config['init_rbf']
        self.s_dims = torch.tensor(rbf_config['s_dims'])  # ... h w, used for initializing rbf
        self.kp_ref_config = rbf_config['kp_ref_config']
        self.rbf_type = rbf_config['rbf_type']
        self.rbf_lc0_normalize = rbf_config['rbf_lc0_normalize']
        self.n_kernel = rbf_config['n_kernel']
        self.point_nn_kernel = rbf_config['point_nn_kernel']
        ks_alpha = rbf_config['ks_alpha']
        self.lc0_dim = rbf_config['lc0_dim']
        self.lc_init = rbf_config['lc_init']
        self.lcb_init = rbf_config['lcb_init']
        self.kp_init = rbf_config['kp_init']
        self.kpb_init = rbf_config['kpb_init']
        self.rbf_suffixes = rbf_config['rbf_suffixes']
        self.pe_lc0_freq = rbf_config['pe_lc0_freq']
        self.pe_lc0_rbf_freq = rbf_config['pe_lc0_rbf_freq']
        self.pe_lc0_rbf_keep = rbf_config['pe_lc0_rbf_keep']
        if self.pe_lc0_rbf_keep == 'all': self.pe_lc0_rbf_keep = self.lc0_dim

        n_freq_level = self.lc0_dim - self.pe_lc0_rbf_keep
        if n_freq_level > 0 and len(self.pe_lc0_rbf_freq) >= 1:
            if len(self.pe_lc0_rbf_freq) == 1:
                freqs = torch.ones([n_freq_level]) * self.pe_lc0_rbf_freq[0]
            elif len(self.pe_lc0_rbf_freq) >= 2:
                amin = np.log2(self.pe_lc0_rbf_freq[0])
                amax = np.log2(self.pe_lc0_rbf_freq[1])
                freqs = torch.linspace(amin, amax, n_freq_level)
                freqs = torch.exp2(freqs)
            self.register_buffer('freqs', freqs)
        else:
            self.freqs = None

        self.pe_kp_freq = rbf_config['pe_kp_freq']
        fix_params = rbf_config['fix_params']
        self.kc_init_config = rbf_config['kc_init_config']
        self.kw_init_config = rbf_config['kw_init_config']
        self.kc_mult = 1
        self.kc_init_regular = {}
        for k, v in self.kc_init_config.items():
            if v['type'] == 'none':
                self.kc_init_regular[k] = True
            else:
                self.kc_init_regular[k] = False
        sparse_embd_grad = False
        self.rbf_train_start = False

        n_params_ref = 0
        for i, res in enumerate(self.kp_ref_config['multiscale_res']):
            n_params_ref += int(self.kp_ref_config['resolution'][0] * res)**2 * 3 * \
                self.kp_ref_config['output_coordinate_dim'][i]

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.kp_dim = 0
        for i, res in enumerate(self.multiscale_res_multipliers):
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                int(r * res) for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"][i],
                reso=config["resolution"],
                a=self.kp_init[0],
                b=self.kp_init[1],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.kp_dim += gp[-1].shape[1]
            else:
                self.kp_dim = gp[-1].shape[1]
            self.grids.append(gp)
        log.info(f"Initialized model grids: {self.grids}")
        n_params_kplanes = int(util_misc.count_parameters(self.grids)[-1])
        self.feature_dim = self.kp_dim

        # 2. Init rbf
        self.kc0 = nn.ModuleList()
        self.ks0 = nn.ModuleList()
        self.lc0 = nn.ModuleList()
        if self.n_kernel == 'auto':
            n_params_rbf = n_params_ref - n_params_kplanes
            self.n_kernel = int(n_params_rbf // (self.lc0_dim + util_misc.get_rbf_params_per_kernel(
                self.rbf_type, self.in_dim, self.kc_mult)))
        if self.n_kernel > 0 and self.lc0_dim > 0 and not self.get_init_data:
            if self.kc_init_regular['0']:
                self.n_kernel = util_misc.get_lower_int_power(self.n_kernel, self.in_dim)
            print('n_kernel:', self.n_kernel)
            self.lc_dims = [[self.n_kernel, self.lc0_dim]]
            self.kc0, self.ks0, self.lc0, self.ks_dims, k_dims, kc_interval = self.create_rbf_params(
                self.rbf_type, self.n_kernel, self.in_dim, self.lc0_dim, sparse_embd_grad, self.cmin, self.cmax, ks_alpha, is_bag=False)
            nn.init.uniform_(self.lc0.weight, self.lc_init[0], self.lc_init[1])
            self.register_buffer(f'k_dims_0', k_dims)
            self.register_buffer(f'kci0', kc_interval)
            self.rbf_fn = eval(f'util_network.rbf_{self.rbf_type}_fb')
            if self.init_rbf:
                self.init_rbf_params(init_data)
            self.rbf_train_start = True
            self.feature_dim += self.lc0_dim * (len(self.pe_lc0_freq) + 1)

        self.feature_dim *= len(self.pe_kp_freq) + 1

        # kpb0
        if self.kpb_init is not None:
            self.kpb = torch.nn.Parameter(torch.zeros(self.feature_dim))
            nn.init.uniform_(self.kpb, self.kpb_init[0], self.kpb_init[1])

        # 3. Init appearance code-related parameters
        self.use_average_appearance_embedding = True  # for test-time
        self.use_appearance_embedding = use_appearance_embedding
        self.num_images = num_images
        self.appearance_embedding = None
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = nn.Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.appearance_embedding_dim = 0

        # 4. Init decoder params
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        # 4. Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB
            # This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.in_dim_color = (
                    self.direction_encoder.n_output_dims
                    + self.geo_feat_dim
                    + self.appearance_embedding_dim
            )
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

        # Fix params
        util_network.fix_params(self, fix_params)

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
        t = time.time()
        pts, _, _ = util_misc.get_grid_points(self.s_dims.tolist(), align_corners=True, 
            vmin=self.cmin.tolist(), vmax=self.cmax.tolist(), device=0)
        pts.flip(-1)  # x y z
        print('Generate grid points:', time.time() - t)

        # Get density and features at these points
        if self.get_init_data:
            t = time.time()
            with torch.no_grad():
                chunk_size = 2**21
                density_all = []
                features_all = []
                for i in range(0, pts.shape[0], chunk_size):
                    features = interpolate_ms_features(
                        pts[i:i+chunk_size], ms_grids=self.grids,
                        grid_dimensions=self.grid_config[0]["grid_dimensions"],
                        concat_features=self.concat_features, num_levels=None)

                    if self.linear_decoder:
                        density_before_activation = self.sigma_net(features)  # [batch, 1]
                    else:
                        features = self.sigma_net(features)  # float16
                        features, density_before_activation = torch.split(
                            features, [self.geo_feat_dim, 1], dim=-1)  # float16

                    density = self.density_activation(density_before_activation.to(pts))  # float32
                    
                    density_all.append(density.cpu())
                    features_all.append(features.cpu())
                density_all = torch.cat(density_all, 0)
                features_all = torch.cat(features_all, 0)
                del density, features
                torch.cuda.empty_cache()
            print('Get density and features at grid points:', time.time() - t)
            density_all, features_all = density_all.cpu(), features_all.cpu()
            init_data = {'density': density_all, 'features': features_all}

            # # Save init data
            # os.makedirs(os.path.dirname(self.init_data_fp), exist_ok=True)
            # torch.save(init_data, self.init_data_fp)
            # density_all = density_all.reshape(*self.s_dims.tolist(), -1)
            # vertices, triangles = mcubes.marching_cubes(density_all.numpy()[..., 0], 50.)
            # trimesh.Trimesh(vertices, triangles, process=False).export(self.init_data_fp.split('.')[0] + '.ply')
            
            torch.cuda.empty_cache()
            return init_data

        density_all = init_data['density']
        features_all = init_data['features'].to(torch.float32)
        del init_data

        # Init rbf parameters
        util_network.init_nerf_rbf_params(self, pts, density_all, features_all, self.kc_init_config, 
                                          self.kw_init_config, device=0)
        del density_all, features_all
        torch.cuda.empty_cache()

        # Build kd tree
        self.kdtree0 = KDTree(getattr(self, 'kc0').weight.detach().cpu().numpy())

    def step_after_iter(self, step):
        if step == self.init_steps and self.get_init_data:
            return self.init_rbf_params()

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
            kc = getattr(self, 'kc' + suffix).weight[None]  # [1 k d]
            ks = getattr(self, 'ks' + suffix).weight[None]
            ks = ks.view(*ks.shape[:2], *self.ks_dims)  # [1 k d d] or [1 k 1]
            rbf_out, _ = self.rbf_fn(x_g, kc, ks)  # [p k_topk]
        else:
            kc = getattr(self, 'kc' + suffix)(kernel_idx)  # [p k d]
            ks = getattr(self, 'ks' + suffix)(kernel_idx).view(*kernel_idx.shape, *self.ks_dims)  # [p k d d] or [p k 1]
            rbf_out, _ = self.rbf_fn(x_g, kc, ks)  # [p k_topk]
        return rbf_out

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])  # [n_rays * n_samples, 3]
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)

        # Forward rbf lc0
        if len(features) >= 1 and self.rbf_train_start:
            suffix = '0'
            if self.point_nn_kernel <= 0:  # Use all kernels for each point
                rbf_out = self.forward_rbf(pts, None, suffix)  # [p nk]
                if self.rbf_lc0_normalize:
                    rbf_out = rbf_out / (rbf_out.detach().sum(-1, keepdim=True) + 1e-8)
                out = rbf_out @ self.lc0.weight  # [p hfl]
            else:
                kernel_idx = self.forward_kernel_idx(pts, None, suffix)
                rbf_out = self.forward_rbf(pts, kernel_idx, suffix)  # [p k_topk]
                if self.rbf_lc0_normalize:
                    rbf_out = rbf_out / (rbf_out.detach().sum(-1, keepdim=True) + 1e-8)
                out = self.lc0(kernel_idx)  # [p k_topk d_lc0]
                rbf_out = rbf_out[..., None]  # [p k_topk 1]
                if self.freqs is not None:
                    if self.pe_lc0_rbf_keep > 0:
                        rbf_out = torch.cat([rbf_out.expand(-1, -1, self.pe_lc0_rbf_keep), 
                            torch.sin(rbf_out * self.freqs[None, None])], -1)  # [p k_topk d_lc0]
                    else:
                        rbf_out = torch.sin(rbf_out * self.freqs[None, None])  # [p k_topk d_lc0]
                out = (out * rbf_out).sum(1)  # [p d_lc0]

            features = torch.cat([features, out], -1)

        if len(self.pe_kp_freq) > 0:
            features = torch.cat([features] + [features*i for i in self.pe_kp_freq], -1)
        if self.kpb_init is not None:
            features = features + self.kpb[None]
        features = torch.sin(features)

        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(
                features, [self.geo_feat_dim, 1], dim=-1)

        density = self.density_activation(
            density_before_activation.to(pts)
        ).view(n_rays, n_samples, 1)
        return density, features

    def forward(self,
                pts: torch.Tensor,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        camera_indices = None
        if self.use_appearance_embedding:
            if timestamps is None:
                raise AttributeError("timestamps (appearance-ids) are not provided.")
            camera_indices = timestamps
            timestamps = None
        density, features = self.get_density(pts, timestamps)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)

        if self.linear_decoder:
            color_features = [features]
        else:
            color_features = [encoded_directions, features.view(-1, self.geo_feat_dim)]

        if self.use_appearance_embedding:
            if camera_indices.dtype == torch.float32:
                # Interpolate between two embeddings. Currently they are hardcoded below.
                #emb1_idx, emb2_idx = 100, 121  # trevi
                emb1_idx, emb2_idx = 11, 142  # sacre
                emb_fn = self.appearance_embedding
                emb1 = emb_fn(torch.full_like(camera_indices, emb1_idx, dtype=torch.long))
                emb1 = emb1.view(emb1.shape[0], emb1.shape[2])
                emb2 = emb_fn(torch.full_like(camera_indices, emb2_idx, dtype=torch.long))
                emb2 = emb2.view(emb2.shape[0], emb2.shape[2])
                embedded_appearance = torch.lerp(emb1, emb2, camera_indices)
            elif self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                if hasattr(self, "test_appearance_embedding"):
                    embedded_appearance = self.test_appearance_embedding(camera_indices)
                elif self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.appearance_embedding.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

            # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
            ea_dim = embedded_appearance.shape[-1]
            embedded_appearance = embedded_appearance.view(-1, 1, ea_dim).expand(n_rays, n_samples, -1).reshape(-1, ea_dim)
            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)

        if self.linear_decoder:
            if self.use_appearance_embedding:
                basis_values = self.color_basis(torch.cat([directions, embedded_appearance], dim=-1))
            else:
                basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)

        return {"rgb": rgb, "density": density}

    def get_params(self):
        lc0_params = {k: v for k, v in self.lc0.named_parameters(prefix="lc0")}
        kc0_params = {k: v for k, v in self.kc0.named_parameters(prefix="kc0")}
        ks0_params = {k: v for k, v in self.ks0.named_parameters(prefix="ks0")}
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys() and 
            k not in lc0_params.keys() and k not in kc0_params.keys() and k not in ks0_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
            "lc0": list(lc0_params.values()),
            "kc0": list(kc0_params.values()),
            "ks0": list(ks0_params.values()),
        }
