import torch
import numpy as np
import skimage
from scipy.spatial import cKDTree as KDTree
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import lpips


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    """
    Modified from https://github.com/kwea123/MINER_pl/blob/84c089f097890a13b59d5d4ca17ca79f39d707e0/metrics.py#L5
    """
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return value.mean()
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean', vmin=0, vmax=1):
    """
    Modified from https://github.com/kwea123/MINER_pl/blob/84c089f097890a13b59d5d4ca17ca79f39d707e0/metrics.py#L14
    """
    if torch.is_tensor(image_pred):
        return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))
    elif isinstance(image_pred, np.ndarray):
        return -10*np.log10(mse(image_pred, image_gt, valid_mask, reduction))


def mse2psnr(x, vmax=1): return 20*np.log10(vmax) - 10*np.log10(x)


def mae(pred, gt):
    return (pred - gt).abs().mean()


def ssim_ski_func(pred, gt, vmin=0, vmax=1):
    if type(gt) is torch.Tensor:
        gt = gt.cpu().numpy()
    if type(pred) is torch.Tensor:
        pred = pred.cpu().numpy()
    return skimage.metrics.structural_similarity(gt, np.clip(pred, a_min=vmin, a_max=vmax), data_range=vmax - vmin, 
        channel_axis=2)


def ssim_func(rgb, gts):
    """
    Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    """
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    max_val = 1.0
    rgb = rgb.cpu().numpy()
    gts = gts.cpu().numpy()
    assert len(rgb.shape) == 3
    assert rgb.shape[-1] == 3
    assert rgb.shape == gts.shape
    import scipy.signal

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(rgb)
    mu1 = filt_fn(gts)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgb**2) - mu00
    sigma11 = filt_fn(gts**2) - mu11
    sigma01 = filt_fn(rgb * gts) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    return np.mean(ssim_map)


ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)


def msssim(rgb, gts):
    """
    Modified from https://github.com/sarafridov/K-Planes/blob/7e3a82dbdda31eddbe2a160bc9ef89e734d9fc54/plenoxels/ops/image/metrics.py#L119
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(torch.permute(rgb[None, ...], (0, 3, 1, 2)),
                   torch.permute(gts[None, ...], (0, 3, 1, 2))).item()


__LPIPS__ = {}


def init_lpips(net_name, device):
    """
    Modified from https://github.com/sarafridov/K-Planes/blob/7e3a82dbdda31eddbe2a160bc9ef89e734d9fc54/plenoxels/ops/image/metrics.py#L128
    """
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(rgb, gts, net_name='alex', device='cpu'):
    """
    Modified from https://github.com/sarafridov/K-Planes/blob/7e3a82dbdda31eddbe2a160bc9ef89e734d9fc54/plenoxels/ops/image/metrics.py#L132
    """
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gts = gts.permute([2, 0, 1]).contiguous().to(device)
    rgb = rgb.permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gts, rgb, normalize=True).item()


def iou(occ1, occ2, thres):
    """
    Modified from https://github.com/kwea123/MINER_pl/blob/84c089f097890a13b59d5d4ca17ca79f39d707e0/metrics.py#L21
    """
    occ1 = occ1>=thres
    occ2 = occ2>=thres
    area_union = (occ1 | occ2).sum()
    area_intersect = (occ1 & occ2).sum()
    return area_intersect/(area_union+1e-8)


def mesh_metrics(pred_mesh, gt_mesh, n_surface_samples, fscore_tau):
    if pred_mesh.faces.shape[0] == 0:
        return {}

    pred_points, pred_indices = pred_mesh.sample(n_surface_samples, return_index=True)
    pred_points = pred_points.astype(np.float32)
    pred_normals = pred_mesh.face_normals[pred_indices]
    gt_points, gt_indices = gt_mesh.sample(n_surface_samples, return_index=True)
    gt_points = gt_points.astype(np.float32)
    gt_normals = gt_mesh.face_normals[gt_indices]

    kdtree = KDTree(gt_points)
    dist_p2g, indices_p2g = kdtree.query(pred_points)
    kdtree = KDTree(pred_points)
    dist_g2p, indices_g2p = kdtree.query(gt_points)

    out = {}
    out['cd_l1'] = (np.mean(dist_p2g) + np.mean(dist_g2p))
    out['cd_l2'] = (np.mean(dist_p2g**2) + np.mean(dist_g2p**2))

    for tau in fscore_tau:
        precision = np.mean((dist_p2g <= tau).astype(np.float32)) * 100.0
        recall = np.mean((dist_g2p <= tau).astype(np.float32)) * 100.0
        fs = (2 * precision * recall) / (precision + recall + 1e-9)
        out[f'fs_{tau:.0e}'] = fs

    normals_p2g = gt_normals[indices_p2g]
    nc_p2g = np.abs(np.sum(normals_p2g * pred_normals, axis=1))
    normals_g2p = pred_normals[indices_g2p]
    nc_g2p = np.abs(np.sum(normals_g2p * gt_normals, axis=1))
    out['nc'] = 0.5 * (np.mean(nc_p2g) + np.mean(nc_g2p))

    nc_p2g = np.degrees(np.arccos(nc_p2g))
    nc_p2g = np.degrees(np.arccos(nc_g2p))
    out['nae'] = 0.5 * (np.mean(nc_p2g) + np.mean(nc_g2p))

    return out
