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

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--path", type=str, default='./data/sdf/armadillo_nrml.obj')
parser.add_argument("--alias", type=str)
parser.add_argument("--ds_device", type=str, default='auto')
parser.add_argument("--log2_hashmap_size_ref", type=int, default=15)
config = parser.parse_args()
config.config_fp = __file__

if config.alias is None:
    config.alias = config.path.split('/')[-1].split('.')[0].split('_nrml')[0]

config.task = 'sdf'
config.workspace = f'log/sdf'
config.test = False
config.seed = None
config.cfactor = 128
config.train_shuffle_mode = 1
config.train_num_samples = 2**18
config.train_presample = True
config.train_epoch_size = 1000
config.val_resolution = 1024
config.clip_sdf = None
config.record_training = False

config.pe_lc0_freq = [30, 300]
config.pe_lc0_rbf_freq = [2**0, 2**3]
config.num_levels = 16
config.level_dim = 1

# Model
config.arch = 'rbf'
config.rbf_type = 'nlin_s'
config.rbf_lc0_normalize = True
config.n_kernel = 'auto'
config.point_nn_kernel = 8
config.ks_alpha = 1
config.n_hidden_fl = 16

config.num_levels_ref = 16
config.level_dim_ref = 2
config.base_resolution_ref = 16

config.log2_hashmap_size = config.log2_hashmap_size_ref
config.base_resolution = 16
config.desired_resolution = 2048
config.levels_omit = []

config.pe_freqs = []
config.pe_hg0_freq = []
config.pe_lc0_rbf_keep = 0

config.num_layers = 3
config.hidden_dim = 64
config.lc_act = 'none'
config.act = 'relu'
config.lc_init = [1e-4]
config.lcb_init = [1e-4]
config.w_init = [None, None, None]
config.b_init = [None, None, None]
config.a_init = [9, 30, 30, 30]

config.rbf_suffixes = ['0']

config.fix_params = ['kc0', 'ks0']
config.fix_params += ['a0', 'a1', 'a2', 'a3']

# Special
config.ema_decay = None
config.fp16 = False

# Training
config.max_steps = 20000
config.val_freq = 1.0
config.val_first = False
config.log_train = True
config.log_img = False
config.log_kernel_img = False
config.log_sdf_slice = False
config.save_pred = True
config.save_pred_pt = False
config.save_ckpt = False
config.train_metric_list = ['psnr', 'mae']
config.val_metric_list = ['psnr', 'iou', 'mae', 'mesh']
config.metric_config = {}
config.metric_config['n_surface_samples'] = 1000000
config.metric_config['fscore_tau'] = [1e-2, 2e-3, 1e-3, 2e-4, 1e-4]

# Optimizers
eps = 1e-15
lr = 1e-4
config.optims = {}
config.optims["dec"] = {'type': 'Adam', 'lr': 1e-4, 'betas': (0.9, 0.99), 'eps': eps, 'wd': 0}
config.optims["hg0"] = {'type': 'Adam', 'lr': lr, 'betas': (0.9, 0.99), 'eps': eps, 'wd': 0}
config.optims["lc0"] = {'type': 'Adam', 'lr': lr, 'betas': (0.9, 0.99), 'eps': eps, 'wd': 0}
config.optims["lcb0"] = {'type': 'Adam', 'lr': lr, 'betas': (0.9, 0.99), 'eps': eps, 'wd': 0}

# lr schedulers
T_max = config.max_steps
lr_gamma = 10
config.lr_schs = {}
config.lr_schs["dec"] = {'type': 'cosine', 'T_max': T_max, 'gamma': lr_gamma}
config.lr_schs["hg0"] = {'type': 'cosine', 'T_max': T_max, 'gamma': lr_gamma}
config.lr_schs["lc0"] = {'type': 'cosine', 'T_max': T_max, 'gamma': lr_gamma}
config.lr_schs["lcb0"] = {'type': 'cosine', 'T_max': T_max, 'gamma': lr_gamma}

# RBF params init
config.kc_init_config = {}
config.kc_init_config['0'] = {'type': 'v3', 'points_sampling': 1, 'reg_sampling': 0, 'weight_exp': 1, 'weight_thres': 0, 'n_iter': 10, 'weight_src': 'gt_inv', 'weight_src_thres': 1e-9, 'weight_src_npts': 26214400}

config.kw_init_config = {}
config.kw_init_config['0'] = {'type': 'v3', 'points_sampling': 1, 'alpha': 0.5, 'weight_exp': 1, 'weight_thres': 0, 'weight_src': 'gt_inv', 'weight_src_thres': 1e-9, 'weight_src_npts': 26214400}

config.save_fn = f'{config.alias}-{config.arch}'

config.exp_name = f'{config.alias}-{config.arch}'
