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

config = dict(
    dataset_name = 'blender',
    datadir = './data/nerf_synthetic',
    basedir = './log/nerf_synthetic_3m',
    expname = 'rbf_tensorf',
    add_version_id = True,

    data_name = 'chair',
    # data_name = 'drums',
    # data_name = 'ficus',
    # data_name = 'hotdog',
    # data_name = 'lego',
    # data_name = 'materials',
    # data_name = 'mic',
    # data_name = 'ship',

    n_iters = 30000,
    batch_size = 4096,
    batch_size_init = 4096,

    N_voxel_init = int(128**3),
    N_voxel_final = int(200**3),
    N_voxel_real_init = int(128**3),
    N_voxel_real_final = int(235**3),
    upsamp_list = [2000,3000,4000,5500,7000],
    update_AlphaMask_list = [2000,4000],
    shrink_0 = True,
    no_upsample = False,
    scale_reso = True,
    fp16 = True,

    N_vis = 5,
    vis_every = 10000,

    render_test = 1,
    save_img = True,
    save_video = True,
    save_ckpt = True,

    model_name = 'nerf_tensorf.network.TensorVMSplitRBF',
    resol_min = 16,
    n_level = 2,
    level_types = ['vm'],
    n_lamb_sigma = [6, 16],
    n_lamb_sh = [12, 32],
    data_dim_color = None,  # None means no basis_mat

    shadingMode = 'ASG_Fea',
    fea2denseAct = 'softplus',

    view_pe = 3,
    fea_pe = -1,
    btn_freq = [3e-1, 1e1],
    featureC = 256,

    lr_basis = 1e-3,
    lr_basis_mat = 1e-3,
    lr_init = 0.02,
    lr_g_factor = 0.02,

    Ro_weight = 0.3,
    L1_weight_inital = 8e-5 / 6,
    L1_weight_rest = 0.,
    Ortho_weight = 0.,
    TV_weight_density = 0.,
    TV_weight_app = 0.,
    rm_weight_mask_thre = 1e-4,
)

# RBF related
config['rbf_config'] = {
        'get_init_data': False,
        'init_steps': 1000,
        'init_data_fp': f"init_data/data_name",
        'init_rbf': True,
        's_dims': 'aabb',
        'ref_config': {
          'N_voxel_final': int(128**3),
          'n_lamb_sigma': [16],
          'n_lamb_sh': [48],
        },
        'rbf_type': 'ivq_a',
        'rbf_lc0_normalize': False,
        'n_kernel': 'auto',
        'point_nn_kernel': 5,
        'ks_alpha': 1,
        'lc0_dim': 32,
        'lcd0_dim': 0,
        'pe_lc0_freq': [],
        'pe_lc0_rbf_freq': [],
        'pe_lc0_rbf_keep': 0,
        'lc_init': [-1e-6, 1e-6],
        'lcd_init': [-0e-6, 0e-6],
        'lcb_init': None,
        'rbf_suffixes': ['0'],
        'fix_params': ['kc0', 'ks0'],
        'kc_init_config': {
        '0': {'type': 'v3', 'points_sampling': 1, 'reg_sampling': 0, 'weight_exp': 1, 'weight_thres': 0, 'n_iter': 10,
              'weight_src': 'alpha_feat_grad'},
        },
        'kw_init_config': {
        '0': {'type': 'v3', 'points_sampling': 1, 'alpha': 0.3, 'weight_exp': 1, 'weight_thres': 0,
              'weight_src': 'alpha_feat_grad'},
        },
        'lr_config': {
          'lc0': 2e-2,
          'lcd0': 2e-2,
          'lcb0': 2e-2,
          'kc0': 3e-5,
          'ks0': 1e-2,
        },
    }


import os

# Process config
config['datadir'] = os.path.join(config['datadir'], 'data_name')
config['expname'] = f"data_name_{config['expname']}"

rbf_config = config['rbf_config']
rbf_config['init_data_fp'] += f"_{rbf_config['init_steps']}.pt"

if config['no_upsample']:
  config['N_voxel_real_init'] = config['N_voxel_real_final']
