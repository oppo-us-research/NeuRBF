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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math

import torch
import torch.nn.functional as F

import util_figure


def write_kernel_summary_2d(writer, step, data_gt, data_gt_grad, data_pred, kc, kw_sq, cmin=-1, cmax=1, suffix='', sidelength=None):
    '''
    data_gt: [1 c h w]
    data_gt_grad: [1 c h w]
    data_pred: [1 c h w]
    kc: [p d]
    kw_sq: [p] or [p d] or [p d d]
    '''
    size_scale_factor = (512**2 / torch.tensor(data_gt.shape[2:]).prod()).sqrt()
    data_gt = F.interpolate(data_gt, scale_factor=size_scale_factor, mode='bilinear', align_corners=False, 
        recompute_scale_factor=True)[0, ...].detach().cpu().numpy().transpose([1, 2, 0])
    data_gt_grad = F.interpolate(data_gt_grad, scale_factor=size_scale_factor, mode='bilinear', align_corners=False,
        recompute_scale_factor=True)[0, ...].detach().cpu().numpy().transpose([1, 2, 0])
    data_pred = F.interpolate(data_pred, scale_factor=size_scale_factor, mode='bilinear', align_corners=False,
        recompute_scale_factor=True)[0, ...].detach().cpu().numpy().transpose([1, 2, 0])

    vmin, vmax = 0, 1
    cmin, cmax = cmin[None, :].astype(np.float32), cmax[None, :].astype(np.float32)
    dims = data_gt.shape[:2][::-1]  # w h

    scale_factor = (np.array(dims, dtype=np.float32)[None, ...] / (cmax - cmin))  # [1 d]
    kc = (kc - cmin) * scale_factor  # Scale centers to [0, dims]
    kc -= 0.5  # The center of top-left pixel is [0, 0]
    if kw_sq.ndim == 1:
        kw_sq = kw_sq[:, None] * scale_factor ** 2
    elif kw_sq.ndim == 2:
        kw_sq = kw_sq * scale_factor ** 2
    elif kw_sq.ndim == 3:
        scale_factor_mat = scale_factor.T * scale_factor
        kw_sq = kw_sq * scale_factor_mat[None, ...]
    else:
        raise NotImplementedError

    xlim_scaled = (-dims[0] * 0.1, dims[0] * 1.1)
    ylim_scaled = (-dims[1] * 0.1, dims[1] * 1.1)

    fig, axes = plt.subplots(1, 1, figsize=[5*1, 5])
    axes = [axes]
    axes[0].imshow(data_gt_grad)
    for i in [0]:
        axes[i] = util_figure.plot_ellipses_v1(axes[i], kc, kw_sq, xlim_scaled, ylim_scaled, alpha=0.2, 
            flip_angle=True)
        axes[i].invert_yaxis()
    if fig is not None:
        writer.add_figure(f'kernels{suffix}', fig, global_step=step)


def write_image_summary_new(writer, step, data_gt, data_gt_grad, data_pred, cmin=-1, cmax=1, suffix=''):
    '''
    data_gt: [1 c h w]
    data_gt_grad: [1 c h w]
    data_pred: [1 c h w]
    '''
    size_scale_factor = (512**2 / torch.tensor(data_gt.shape[2:]).prod()).sqrt()
    data_gt = F.interpolate(data_gt, scale_factor=size_scale_factor, mode='bilinear', align_corners=False, 
        recompute_scale_factor=True)[0, ...].detach().cpu()
    data_pred = F.interpolate(data_pred, scale_factor=size_scale_factor, mode='bilinear', align_corners=False,
        recompute_scale_factor=True)[0, ...].detach().cpu()

    vmin, vmax = 0, 1

    writer.add_images(f'out{suffix}',
                    torch.cat([
                        data_gt.clip(vmin, vmax),
                        data_pred.clip(vmin, vmax),
                        ((data_gt - data_pred).abs()*50).clip(vmin, vmax),
                        ((data_gt - data_pred).pow(2).sum(0, keepdim=True).sqrt().expand(data_gt.shape[0], -1, -1)*50).clip(vmin, vmax),
                    ], -1),
                    global_step=step, dataformats='CHW')


def write_sdf_slice_summary(writer, step, data_gt, data_pred, suffix=''):
    '''
    data_gt: [1 c d h w]
    data_pred: [1 c d h w]
    '''
    size_scale_factor = (512**2 / torch.tensor(data_gt.shape[-2:]).prod()).sqrt()
    slice_id = math.ceil(data_gt.shape[-1]*1/2) - 1
    data_gt = torch.cat([data_gt[..., slice_id, :, :], data_gt[..., slice_id, :], data_gt[..., slice_id]], -2)  # 1 c 3h w
    data_pred = torch.cat([data_pred[..., slice_id, :, :], data_pred[..., slice_id, :], data_pred[..., slice_id]], -2)  # 1 c 3h w

    data_gt = F.interpolate(data_gt, scale_factor=size_scale_factor, mode='bilinear', align_corners=False, 
        recompute_scale_factor=True)[0, ...].detach().cpu().permute(1, 2, 0)  # 3h w c
    data_pred = F.interpolate(data_pred, scale_factor=size_scale_factor, mode='bilinear', align_corners=False,
        recompute_scale_factor=True)[0, ...].detach().cpu().permute(1, 2, 0)  # 3h w c

    vmin, vmax = -0.1, 0.1
    fig, axes = plt.subplots(1, 4, figsize=[5*4, 5*3])
    axes[0].imshow(data_gt, vmin=vmin, vmax=vmax)
    axes[0].contour(data_gt[..., 0], levels=[0], colors='k', linewidths=0.5)
    axes[1].imshow(data_pred, vmin=vmin, vmax=vmax)
    axes[1].contour(data_pred[..., 0], levels=[0], colors='k', linewidths=0.5)
    axes[2].imshow((data_gt - data_pred).abs()*10, vmin=vmin, vmax=vmax)
    axes[2].contour(data_gt[..., 0], levels=[0], colors='r', linewidths=0.5, alpha=1)
    axes[2].contour(data_pred[..., 0], levels=[0], colors='k', linewidths=0.5, alpha=1)
    axes[3].imshow((data_gt - data_pred).abs()*10, vmin=vmin, vmax=vmax)
    for ax in axes:
        ax.invert_yaxis()
        ax.axis('off')

    writer.add_figure(f'out{suffix}', fig, global_step=step)


def write_tensor3d_slice_summary(writer, step, data_gt, data, suffix=''):
    '''
    data: [1 c d h w]
    '''
    size_scale_factor = (512**2 / torch.tensor(data.shape[-2:]).prod()).sqrt()
    slice_id = math.ceil(data.shape[-1]*1/2) - 1
    nrow, ncol = 4, 8+1
    fig, axes = plt.subplots(nrow, ncol, figsize=[2.25*ncol, 2.25*nrow])
    
    data_gt = data_gt[..., slice_id, :, :]  # 1 c h w
    data_gt = F.interpolate(data_gt, scale_factor=size_scale_factor, mode='bilinear', align_corners=False, 
        recompute_scale_factor=True)[0, ...].detach().cpu().permute(1, 2, 0)  # 3h w c
    for i in range(nrow):
        axes[i, 0].imshow(data_gt)
        axes[i, 0].invert_yaxis()
        axes[i, 0].axis('off')

    for i in range(nrow):
        for j in range(1, ncol):
            cid = i*(ncol-1)+j-1
            if cid < data.shape[1]:
                data_show = data[:, cid:cid+1, slice_id, :, :]  # 1 1 h w
                data_show = F.interpolate(data_show, scale_factor=size_scale_factor, mode='bilinear', 
                    align_corners=False, recompute_scale_factor=True)[0, ...].detach().cpu().permute(1, 2, 0)  # h w 1
                axes[i, j].imshow(data_show)
            axes[i, j].invert_yaxis()
            axes[i, j].axis('off')

    plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0.03, hspace=0.03)
    writer.add_figure(f'{suffix}', fig, global_step=step)
