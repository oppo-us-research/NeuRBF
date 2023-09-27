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
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection, EllipseCollection
import matplotlib.transforms as transforms
import numpy as np


def plot_ellipses_v1(ax, means, covs, xlim, ylim, alpha=0.5, flip_angle=True):
    '''
    mean: [k d], numpy array
    cov: [k] or [k d] or [k d d], numpy array
    '''
    if covs.ndim == 1:
        covs = covs[..., None, None] * np.eye(2)[None, ...]
    elif covs.ndim == 2:
        covs = covs[..., None] * np.eye(2)[None, ...]
    elif covs.ndim > 3:
        raise NotImplementedError

    a = covs[:, 0, 0]
    b = covs[:, 0, 1]
    c = covs[:, 1, 1]
    temp1 = (a + c) / 2
    temp2 = np.sqrt(((a - c) / 2) ** 2 + b ** 2)
    lambda1 = temp1 + temp2
    lambda2 = temp1 - temp2
    lambda2[lambda2 < 0] = 0

    theta = np.degrees(np.arctan2(lambda1 - a, b))
    theta[(b == 0) * (a >= c)] = 0
    theta[(b == 0) * (a < c)] = 90
    if flip_angle: theta = -theta

    col = EllipseCollection(np.sqrt(lambda1)*2, np.sqrt(lambda2)*2, theta, units='x', offsets=means, 
        offset_transform=ax.transData, alpha=alpha, facecolor='w', edgecolor='y')
    ax.add_collection(col)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=10)

    return ax


def plot_ellipse(ax, mean, cov, **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson) if pearson <= 1 else 0
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = np.sqrt(cov[0, 0])
    scale_y = np.sqrt(cov[1, 1])
    transf = transforms.Affine2D().rotate_deg(-45).scale(scale_x, scale_y).translate(mean[0], mean[1])
    ellipse.set_transform(transf + ax.transData)
    
    ax.add_patch(ellipse)

    return ax


def plot_ellipses(ax, means, covs, xlim, ylim, alpha=0.5):
    if covs.ndim == 1:
        covs = covs[..., None, None] * np.eye(2)[None, ...]
    elif covs.ndim == 2:
        covs = covs[..., None] * np.eye(2)[None, ...]
    elif covs.ndim > 3:
        raise NotImplementedError

    for i in range(0, means.shape[0]):
        ax = plot_ellipse(ax, means[i, ...], covs[i, ...], alpha=alpha, facecolor='w', edgecolor='y')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=10)

    return ax
