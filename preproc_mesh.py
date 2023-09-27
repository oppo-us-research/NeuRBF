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

import trimesh
import pysdf
import numpy as np
import torch
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--path", type=str)
config = parser.parse_args()

path = config.path

print(path)
mesh = trimesh.load(path, force='mesh')
if not mesh.is_watertight:
    print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")

# Normalize to [-1, 1] (different from instant-sdf where is [0, 1])
vs = mesh.vertices
vmin = vs.min(0)
vmax = vs.max(0)
v_center = (vmin + vmax) / 2
vs = (vs - v_center[None, :])
v_scale = 1. / (np.sqrt(np.sum(vs**2, -1).max()) / 0.99)
vs *= v_scale
mesh.vertices = vs
mesh.export(path.rsplit('.', 1)[0] + '_nrml.obj')

# Generate evaluation points
sdf_fn = pysdf.SDF(mesh.vertices, mesh.faces)
ns_taus = []
n_points = {f'ns_{ns_tau:.0e}': int(1e7) for ns_tau in ns_taus}
n_points.update({'unif': int(1e7)})
points = {}
sdfs = {}
# surface
if 'surf' in n_points:
    points['surf'] = mesh.sample(n_points['surf']).view(np.ndarray)
    sdfs['surf'] = np.zeros([points['surf'].shape[0], 1])
# near surface
for ns_tau in ns_taus:
    k = f'ns_{ns_tau:.0e}'
    points[k] = mesh.sample(n_points[k]).view(np.ndarray)
    points[k] += ns_tau * np.random.rand(*points[k].shape)
# uniform random
if 'unif' in n_points:
    points['unif'] = (np.random.rand(n_points['unif'], 3) * 2 - 1)
# Compute SDF
for k in points:
    if k != 'surf':
        sdfs[k] = -sdf_fn(points[k])[:, None]
    points[k] = points[k].astype(np.float32)
    sdfs[k] = sdfs[k].astype(np.float32)
torch.save({'points': points, 'sdfs': sdfs}, path.rsplit('.', 1)[0] + '_nrml_eval_points.pt')
