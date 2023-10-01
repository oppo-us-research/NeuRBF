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
import numpy as np
import kornia
import math
import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import trimesh
import pysdf

import util_misc


class IMGDataset(Dataset):
    def __init__(self, gt, cmin, cmax, s_dims, num_samples=2**18, ns_per_block=1, shuffle_mode=1, device=0):
        super().__init__()
        self.gt = gt
        if self.gt.dtype is torch.uint8:
            self.gt_cpu = self.gt.to(dtype=torch.float32) / 255.  # [h w c]
        else:
            raise NotImplementedError
        self.device = device
        self.device_cuda = 0 if device == 'cpu' else device
        self.shuffle_mode = shuffle_mode

        if not isinstance(cmin, list):
            cmin = [cmin] * self.s_dims.shape[0]
        if not isinstance(cmax, list):
            cmax = [cmax] * self.s_dims.shape[0]
        self.cmin = torch.tensor(cmin)  # ... y x
        self.cmax = torch.tensor(cmax)  # ... y x
        self.s_dims = torch.tensor(s_dims)  # ... h w
        self.n_point = self.s_dims.prod().item()
        self.d = self.s_dims.shape[0]

        if self.shuffle_mode in [0, 1]:
            self.num_samples = num_samples
            self.size = math.ceil(self.n_point / self.num_samples)
            self.b_dims = torch.ones_like(self.s_dims)
            self.p_dims = self.s_dims.clone()
        elif self.shuffle_mode in [2]:
            self.ns_per_block = ns_per_block
            n_blocks = math.ceil(num_samples / self.ns_per_block)
            self.p_dims = ((self.s_dims.prod() / n_blocks)**(1 / self.d)).floor().repeat(self.d).to(torch.long)
            self.b_dims = (self.s_dims / self.p_dims).ceil().to(torch.long)
            self.num_samples = self.ns_per_block * self.b_dims.prod().item()
            self.size = math.ceil(self.p_dims.prod().item() / self.ns_per_block)
        else:
            raise NotImplementedError

        # get spatial gradients
        self.gt_grad = kornia.filters.SpatialGradient(mode='sobel', order=1, normalized=True)(self.gt_cpu.movedim(-1, 0)[None])  # [1 c 2 h w]
        self.gt_grad = self.gt_grad[0].movedim((0, 1), (-2, -1)).pow(2).sum(dim=[-2, -1]).sqrt()[..., None]  # [h w 1]
        self.gt_grad_cpu = self.gt_grad  # [h w 1]

        # reshape
        self.gt = self.gt.reshape(-1, self.gt.shape[-1])
        self.gt_grad = self.gt_grad.reshape(-1, self.gt_grad.shape[-1])

        # get points
        self.points = util_misc.get_grid_points(self.s_dims.tolist(), align_corners=False, 
            vmin=self.cmin.tolist(), vmax=self.cmax.tolist())[0].flip(-1)  # (... * h * w, d), x y ...
    
    def __len__(self):
        return self.size

    def prepare(self):
        self.points = self.points.to(self.device_cuda)
        self.gt = self.gt.to(self.device_cuda)

        if self.shuffle_mode > 0:
            self.generate_point_order(True)
        else:
            self.generate_point_order(False)

    def generate_point_order(self, shuffle=False):
        if shuffle:
            if self.shuffle_mode == 0:
                pass
            elif self.shuffle_mode == 1:
                self.point_order = util_misc.random_shuffle_chunked(self.n_point, chunk_size=10000*10000, device=self.device)
            elif self.shuffle_mode == 2:
                self.point_order, self.point_mask = util_misc.stratified_shuffle_chunked(self.s_dims, self.b_dims, 
                    chunk_size=10000*5000, device=self.device)
            else:
                raise NotImplementedError

            if self.device == 'cpu': torch.cuda.empty_cache()
        else:
            self.point_order = torch.arange(0, self.n_point, dtype=torch.long, device=self.device_cuda)

    def shuffle_order(self, batch_idx):
        if batch_idx == 0 and self.shuffle_mode > 0:
            self.generate_point_order(True)

    def __getitem__(self, idx):
        if self.shuffle_mode < 2:
            point_idx = self.point_order[idx*self.num_samples:(idx+1)*self.num_samples]
        elif self.shuffle_mode == 2:
            mask = self.point_mask[:, idx*self.ns_per_block:(idx+1)*self.ns_per_block].reshape(-1)
            point_idx = self.point_order[:, idx*self.ns_per_block:(idx+1)*self.ns_per_block].reshape(-1)[mask]
        else:
            raise NotImplementedError
        point_idx = point_idx.to(torch.long)

        return {'point_idx': point_idx, 'points': self.points[point_idx], 
            'gt': self.gt[point_idx].to(torch.float32)/255.}


class SDFDataset(Dataset):
    def __init__(self, mesh, cmin, cmax, s_dims, num_samples=2**18, size=100, presample=False, is_grid=False, shuffle_mode=0, clip_sdf=None, mesh_fp='', normalize_mesh=False, device=0):
        super().__init__()
        self.mesh = mesh.copy()
        self.s_dims = torch.tensor(s_dims)  # ... h w
        self.n_point = self.s_dims.prod().item()
        self.size = size
        self.presample = presample
        self.is_grid = is_grid
        self.device = device
        self.device_cuda = device
        self.shuffle_mode = shuffle_mode
        self.ratio = [4, 3, 1]
        self.ratio_total = sum(self.ratio)

        self.d = self.s_dims.shape[0]
        if not isinstance(cmin, list):
            cmin = [cmin] * self.d
        if not isinstance(cmax, list):
            cmax = [cmax] * self.d
        self.cmin = torch.tensor(cmin)  # ... y x
        self.cmax = torch.tensor(cmax)  # ... y x

        if normalize_mesh:
            # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
            vs = self.mesh.vertices
            vmin = vs.min(0)
            vmax = vs.max(0)
            v_center = (vmin + vmax) / 2
            vs = (vs - v_center[None, :])
            v_scale = 1. / (np.sqrt(np.sum(vs**2, -1).max()) / 0.99)
            vs *= v_scale
            self.mesh.vertices = vs

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        if self.is_grid:
            print('Presample grid points and SDF')
            self.presample = True
            self.points = util_misc.get_grid_points(self.s_dims.tolist(), align_corners=True, 
                vmin=self.cmin.tolist(), vmax=self.cmax.tolist())[0].flip(-1)  # (... * h * w, d), x y ...
            self.size = math.ceil(self.points.shape[0] / self.num_samples)
            self.gt = torch.zeros([self.points.shape[0], 1])
            chunk_size = int(2**24)
            for i in tqdm(range(0, self.points.shape[0], chunk_size)):
                self.gt[i:i+chunk_size] = torch.from_numpy(-self.sdf_fn(self.points[i:i+chunk_size].numpy())[:, None].astype(np.float32))
        elif self.presample:
            save_fp = mesh_fp.rsplit('.', 1)[0] + '_train_points.pt'
            if os.path.exists(save_fp):
                print('Load random points and SDF')
                temp = torch.load(save_fp)
                self.points = temp['points']
                self.gt = temp['gt']
                self.size = self.points.shape[0]
                
                if self.points.shape[1] != self.num_samples:
                    num_samples = self.points.shape[1]
                    size_new = []
                    start = 0
                    for i in self.ratio:
                        end = start + int(num_samples * i / self.ratio_total)
                        size_new.append((self.size * (end - start)) // int(self.num_samples * i / self.ratio_total))
                        start = end
                    self.size = min(size_new)
                    
                    self.points = []
                    self.gt = []
                    start = 0
                    for i in self.ratio:
                        end = start + int(num_samples * i / self.ratio_total)
                        num_samples_i = int(self.num_samples * i / self.ratio_total)
                        self.points.append(temp['points'][:, start:end].reshape(-1, 3)[:self.size*num_samples_i, :].reshape(self.size, -1, 3))
                        self.gt.append(temp['gt'][:, start:end].reshape(-1, 1)[:self.size*num_samples_i].reshape(self.size, -1, 1))
                        start = end
                    self.points = torch.cat(self.points, 1)
                    self.gt = torch.cat(self.gt, 1)
            else:
                print('Presample random points and SDF')
                self.points = torch.zeros([self.size, self.num_samples, 3])
                self.gt = torch.zeros([self.size, self.num_samples, 1])
                for i in tqdm(range(self.size)):
                    points, _ = self.sample_points(compute_sdf=False)
                    self.points[i] = torch.from_numpy(points)
                chunk_size = int(2**24) // (self.num_samples * self.ratio[0] // self.ratio_total)
                for i in tqdm(range(0, self.size, chunk_size)):
                    points = self.points[i:i+chunk_size, self.num_samples * self.ratio[0] // self.ratio_total:]
                    npt = points.shape[1]
                    self.gt[i:i+chunk_size, self.num_samples * self.ratio[0] // self.ratio_total:] = \
                        torch.from_numpy(-self.sdf_fn(points.reshape(-1, 3).numpy()).astype(np.float32)).reshape(-1, npt, 1)
                print('Save random points and SDF')
                torch.save({'points': self.points, 'gt': self.gt}, save_fp)

        if self.presample and self.clip_sdf is not None:
            self.gt.clip_(-self.clip_sdf, self.clip_sdf)

        if self.is_grid:
            # get spatial gradients
            self.gt_grad = kornia.filters.SpatialGradient3d(mode='diff', order=1)(self.gt.reshape(*s_dims, -1).movedim(-1, 0)[None])  # [1 c 3 d h w]
            self.gt_grad = self.gt_grad[0].movedim((0, 1), (-2, -1)).pow(2).sum(dim=[-2, -1]).sqrt()[..., None]  # [... h w 1]
            self.gt_grad = self.gt_grad.reshape(-1, self.gt_grad.shape[-1])
            self.gt_grad_cpu = self.gt_grad  # [d h w 1]
        
    def prepare(self):
        if hasattr(self, 'points'):
            self.points = self.points.to(self.device_cuda)
        if hasattr(self, 'gt'):
            self.gt = self.gt.to(self.device_cuda)
            
        if self.shuffle_mode > 0:
            self.generate_point_order(True)
        else:
            self.generate_point_order(False)

    def generate_point_order(self, shuffle=False):
        if self.presample:
            if shuffle:
                if self.shuffle_mode == 0:
                    pass
                elif self.shuffle_mode == 1:
                    if self.is_grid:
                        self.point_order = torch.randperm(self.n_point, dtype=torch.int32, device=self.device)
                    else:
                        self.point_order = torch.randperm(self.size, dtype=torch.int32, device=self.device)
                else:
                    raise NotImplementedError
            else:
                if self.is_grid:
                    pass
                else:
                    self.point_order = torch.arange(0, self.size, dtype=torch.int32, device=self.device_cuda)

    def shuffle_order(self, batch_idx):
        if batch_idx == 0 and self.shuffle_mode > 0:
            self.generate_point_order(True)

    def sample_points(self, compute_sdf=True):
        ratio = self.ratio
        total = self.ratio_total

        # surface
        points_surface = self.mesh.sample(self.num_samples * (ratio[0] + ratio[1]) // total)
        # perturb surface
        points_surface[self.num_samples * ratio[0] // total:] += 0.01 * np.random.randn(self.num_samples * ratio[1] // total, 3)
        # random
        points_uniform = np.random.rand(self.num_samples * ratio[2] // total, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        if compute_sdf:
            sdfs = np.zeros((self.num_samples, 1))
            sdfs[self.num_samples * ratio[0] // total:] = -self.sdf_fn(points[self.num_samples * ratio[0] // total:])[:,None].astype(np.float32)
        else:
            sdfs = None

        return points, sdfs

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.is_grid:
            if not self.presample:
                point_idx = np.zeros([1], dtype=np.int32)
                points, sdfs = self.sample_points()
                if self.clip_sdf is not None:
                    sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)
            else:
                idx_use = self.point_order[idx].to(torch.long)
                point_idx = torch.arange(self.num_samples*idx_use, self.num_samples*(idx_use+1), device=self.points.device)
                points = self.points[idx_use]
                sdfs = self.gt[idx_use]
        else:
            if self.shuffle_mode > 0:
                point_idx = self.point_order[self.num_samples*idx:self.num_samples*(idx+1)].to(torch.long)
            else:
                point_idx = torch.arange(self.num_samples*idx, self.num_samples*(idx+1), device=self.points.device)
            points = self.points[point_idx]
            sdfs = self.gt[point_idx]

        results = {
            'gt': sdfs,
            'points': points,
            'point_idx': point_idx,
        }

        return results
