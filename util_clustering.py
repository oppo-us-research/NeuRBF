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
import cupy as cp
from pykdtree.kdtree import KDTree


class KMeans():
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def init_centers(self, X, n_clusters, sample_weight=None):
        if sample_weight is not None:
            idx = np.random.choice(X.shape[0], size=n_clusters, replace=False, p=sample_weight)
        else:
            idx = np.random.choice(X.shape[0], size=n_clusters, replace=False)
        return X[idx]

    def init_centers_cp(self, X, n_clusters, sample_weight=None):
        if sample_weight is not None:
            idx = cp.random.choice(X.shape[0], size=n_clusters, replace=True, p=sample_weight)
        else:
            idx = cp.random.choice(X.shape[0], size=n_clusters, replace=False)
        return X[idx]

    def compute_centers_loop_np(self, X, labels, sample_weight=None):
        X = X.astype(np.float64)
        sample_weight = sample_weight.astype(np.float64)
        centers = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
        if sample_weight is not None:
            X = X * sample_weight[:, None]
        for k in range(self.n_clusters):
            idx = labels == k
            if idx.sum() == 0:
                centers[k, :] = 0
            else:
                if sample_weight is None:
                    count = idx.sum()
                else:
                    count = sample_weight[idx].sum()
                centers[k, :] = X[idx, :].sum(axis=0) / count
        return centers

    def compute_centers_np(self, X, labels, sample_weight=None):
        ix = np.argsort(labels)
        labels = labels[ix]
        X = X[ix].astype(np.float64)
        if sample_weight is not None:
            sample_weight = sample_weight[ix].astype(np.float64)
            X = X * sample_weight[:, None]
        
        d = np.diff(labels, prepend=0)
        pos = np.flatnonzero(d)
        pos = np.repeat(pos, d[pos])
        pos = np.append(np.insert(pos, 0, 0), len(X))

        X = np.concatenate((np.zeros_like(X[0:1]), X), axis=0)
        X = np.cumsum(X, axis=0)
        if sample_weight is not None:
            sample_weight = np.concatenate((np.zeros_like(sample_weight[0:1]), sample_weight), axis=0)
            sample_weight = np.cumsum(sample_weight, axis=0)

        X = np.diff(X[pos], axis=0)
        if sample_weight is None:
            count = np.diff(pos).clip(min=1)
        else:
            count = np.diff(sample_weight[pos], axis=0)
            count[count==0]=1
        centers = X / count[:, None]

        return centers

    def compute_centers_cupy(self, X, labels, sample_weight=None):
        '''
        X: [p d], cupy float
        labels: [p], cupy int
        sample_weight: [p], cupy float
        '''
        ix = cp.argsort(labels)
        labels = labels[ix]
        X = X[ix]
        if sample_weight is not None:
            sample_weight = sample_weight[ix]
            X = X * sample_weight[:, None]
        
        d = cp.diff(labels, prepend=0)
        pos = cp.flatnonzero(d)
        pos = cp.asarray(np.repeat(pos.get(), d[pos].get()))
        pos = cp.append(cp.concatenate((cp.zeros_like(pos[0:1]), pos)), len(X))

        X = cp.concatenate((cp.zeros_like(X[0:1]), X), axis=0)
        X = cp.cumsum(X, axis=0)
        if sample_weight is not None:
            sample_weight = cp.concatenate((cp.zeros_like(sample_weight[0:1]), sample_weight), axis=0)
            sample_weight = cp.cumsum(sample_weight, axis=0)

        X = cp.diff(X[pos], axis=0)
        if sample_weight is None:
            count = cp.diff(pos)
        else:
            count = cp.diff(sample_weight[pos], axis=0)
        centers = X / count[:, None]

        return centers, count

    def fit(self, X, sample_weight=None, backend=0, gpu=0):
        if backend==0:  # numpy
            self.fit_np(X.cpu().numpy(), sample_weight.cpu().numpy())
        elif backend==1:  # cupy
            with cp.cuda.Device(gpu):
                self.fit_cupy(X, sample_weight)
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        else:
            raise NotImplementedError

    def fit_np(self, X, sample_weight=None, **kwargs):
        self.centers = self.init_centers(X, sample_weight)
        for i in range(self.max_iter):
            centers_old = self.centers
            self.kdtree = KDTree(self.centers)
            _, self.labels = query_chunked(self.kdtree, X, k=1, sqr_dists=True, chunk_size=int(2e8), return_dist=False)
            self.centers = self.compute_centers_np(X, self.labels, sample_weight)
            if np.all(centers_old == self.centers):
                break
        self.kdtree = KDTree(self.centers)
        _, self.labels = query_chunked(self.kdtree, X, k=1, sqr_dists=True, chunk_size=int(2e8), return_dist=False)

    def fit_cupy(self, X, sample_weight=None, **kwargs):
        X_cp = cp.asarray(X, dtype=cp.float64)
        X = X.cpu().numpy().astype(np.float32, copy=False)
        if sample_weight is not None:
            sample_weight_cp = cp.asarray(sample_weight, dtype=cp.float64)
            sample_weight = sample_weight.cpu().numpy().astype(np.float32, copy=False)
            sample_weight_normalized = sample_weight / sample_weight.sum()
        else:
            sample_weight_cp = None
            sample_weight_normalized = None
        centers_cp = self.init_centers(X_cp, self.n_clusters, sample_weight_normalized)
        self.centers = cp.asnumpy(centers_cp).astype(np.float32)
        for i in range(self.max_iter):
            centers_old_cp = centers_cp
            self.kdtree = KDTree(self.centers)
            _, self.labels = query_chunked(self.kdtree, X, k=1, sqr_dists=True, chunk_size=int(2e8), return_dist=False)
            centers_cp, count = reduce_within_clusters_chunked(X_cp, self.n_clusters, cp.asarray(self.labels), 
                sample_weight_cp, chunk_size=int(1e8 / X_cp.shape[-1]))
            centers_cp[count==0] = centers_old_cp[count==0]
            self.centers = cp.asnumpy(centers_cp).astype(np.float32)
            # if cp.all(centers_old_cp == centers_cp):
            #     break
        self.kdtree = KDTree(self.centers)
        _, self.labels = query_chunked(self.kdtree, X, k=1, sqr_dists=True, chunk_size=int(2e8), return_dist=False)

    def predict(self, X, sample_weight=None):
        _, labels = query_chunked(self.kdtree, X.astype(np.float32, copy=False), k=1, sqr_dists=True, 
            chunk_size=int(2e8), return_dist=False)

        return labels


def query_chunked(kd_tree, x, k, sqr_dists, chunk_size=int(2e8), return_dist=False):
    if chunk_size is None: chunk_size = x.shape[0]
    if chunk_size >= x.shape[0]: return kd_tree.query(x, k=k, sqr_dists=sqr_dists)

    dist = np.zeros([x.shape[0], k], dtype=np.float32) if return_dist else None
    idx = np.zeros([x.shape[0], k], dtype=np.uint32)
    if k == 1:
        if return_dist: dist = dist[:, 0]
        idx = idx[:, 0]
    for i in range(0, x.shape[0], chunk_size):
        dist_i, idx[i:i+chunk_size] = kd_tree.query(x[i:i+chunk_size], k=k, sqr_dists=sqr_dists)
        if return_dist: dist[i:i+chunk_size] = dist_i
    return dist, idx


def reduce_within_clusters(X, n_clusters, labels, sample_weight=None, reduce_weight=True):
    '''
    X: [p ...], cupy float
    labels: [p], cupy int
    sample_weight: [p], cupy float
    '''
    X = X.astype(dtype=cp.float64, copy=False)
    sample_weight = sample_weight.astype(dtype=cp.float64, copy=False)

    ix = cp.argsort(labels)
    labels = labels[ix]
    X = X[ix]
    if sample_weight is not None:
        sample_weight = sample_weight[ix]
        X = X * sample_weight.reshape([X.shape[0], *([1]*(X.ndim-1))])
    
    d = cp.diff(labels, prepend=0)
    pos = cp.flatnonzero(d)
    pos = cp.asarray(np.repeat(pos.get(), d[pos].get()))
    pos = cp.append(cp.concatenate((cp.zeros_like(pos[0:1]), pos)), len(X))

    X = cp.concatenate((cp.zeros_like(X[0:1]), X), axis=0)
    X = cp.cumsum(X, axis=0)
    if sample_weight is not None:
        sample_weight = cp.concatenate((cp.zeros_like(sample_weight[0:1]), sample_weight), axis=0)
        sample_weight = cp.cumsum(sample_weight, axis=0)

    X = cp.diff(X[pos], axis=0)
    if sample_weight is None:
        count = cp.diff(pos)
    else:
        count = cp.diff(sample_weight[pos], axis=0)
    if reduce_weight:
        out = X / count.reshape([X.shape[0], *([1]*(X.ndim-1))])
    else:
        out = X

    if out.shape[0] < n_clusters:
        n_fill = n_clusters - out.shape[0]
        out = cp.concatenate([out, cp.zeros([n_fill, *out.shape[1:]])])
        count = cp.concatenate([count, cp.zeros([n_fill])])

    return out, count


def reduce_within_clusters_chunked(X, n_clusters, labels, sample_weight=None, chunk_size=None):
    if chunk_size is None: chunk_size = X.shape[0]
    for i in range(0, X.shape[0], chunk_size):
        out_i, count_i = reduce_within_clusters(X[i:i+chunk_size], n_clusters, labels[i:i+chunk_size], 
            sample_weight[i:i+chunk_size] if sample_weight is not None else None, reduce_weight=False)
        if i == 0:
            out = out_i
            count = count_i
        else:
            out += out_i
            count += count_i
    out /= count.reshape([out.shape[0], *([1]*(out.ndim-1))])
    return out, count
