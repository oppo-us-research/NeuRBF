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

config = {
    "expname": "rbf_kplanes",
    "logdir": "./log/llff",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 4,
    "data_dirs": ["data/nerf_llff_data/fern"],
    # Data settings for LLFF
    "hold_every": 8,
    "contract": False,
    "ndc": True,
    "near_scaling": 0.89,
    "ndc_far": 2.6,

    # Optimization settings
    "num_steps": 40_001,
    "batch_size": 4096,
    "eval_batch_size": 4096,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr_config": {
        "base": 2e-2,
        "lc0": 2e-2,
        "kc0": 3e-5,
        "ks0": 1e-2,
 },

    # Regularization
    "plane_tv_weight": 0e-4,
    "plane_tv_weight_proposal_net": 0e-4,
    "l1_proposal_net_weight": 0,
    "histogram_loss_weight": 1.0, 
    "distortion_loss_weight": 0.001,

    # Training settings
    "train_fp16": True,
    "save_every": 40000,
    "valid_every": 40000,
    "save_outputs": True,

    # Raymarching settings
    "num_samples": 48,
    "single_jitter": False,
    # proposal sampling
    "num_proposal_samples": [256, 128],
    "num_proposal_iterations": 2,
    "use_same_proposal_network": False,
    "use_proposal_weight_anneal": True,
    "proposal_net_args_list": [
        {"resolution": [128, 128, 128], "num_input_coords": 3, "num_output_coords": 8},
        {"resolution": [256, 256, 256], "num_input_coords": 3, "num_output_coords": 8},
    ],

    # Model settings
    "multiscale_res": [1, 2, 4, 8],
    "density_activation": "trunc_exp",
    "concat_features_across_scales": True,
    "linear_decoder": False,
    "grid_config": [{
        "input_coordinate_dim": 3,
        "output_coordinate_dim": [16, 16, 16, 15],
        "grid_dimensions": 2,
        "resolution": [64, 64, 64],
    }],
    # RBF related
 'field_type': 'kplane_rbf',
 'rbf_config': {
    'get_init_data': False,
    'init_steps': 2000,
    'init_data_fp': 'init_data/data_name',
    'init_rbf': True,
    's_dims': [512, 512, 512],
    'kp_ref_config': {
      'multiscale_res': [1, 2, 4, 8],
      'output_coordinate_dim': [16, 16, 16, 16],
      'resolution': [64, 64, 64],
    },
    'rbf_type': 'ivq_a',
    'rbf_lc0_normalize': True,
    'n_kernel': 'auto',
    'point_nn_kernel': 8,
    'ks_alpha': 1,
    'lc0_dim': 16,
    'pe_lc0_freq': [],
    'pe_lc0_rbf_freq': [],
    'pe_lc0_rbf_keep': 'all',
    'pe_kp_freq': [],
    'lc_init': [0.1, 0.5],
    'lcb_init': [0.1, 0.5],
    'kp_init': [0.1, 0.5],
    'kpb_init': [0, 0],
    'rbf_suffixes': ['0'],
    'fix_params': ['kc0', 'ks0'],
    'kc_init_config': {
      '0': {'type': 'v3', 'points_sampling': 1, 'reg_sampling': 0, 'weight_exp': 1, 'weight_thres': 0.5, 'n_iter': 10}
    },
    'kw_init_config': {
      '0': {'type': 'v3', 'points_sampling': 1, 'alpha': 0.5, 'weight_exp': 1, 'weight_thres': 0.5}
    },
 },
}


# Process config
rbf_config = config['rbf_config']
rbf_config['init_data_fp'] += f"_{rbf_config['init_steps']}.pt"
config['num_steps'] -= rbf_config['init_steps']
config['save_every'] -= rbf_config['init_steps']
config['valid_every'] -= rbf_config['init_steps']

config['expname'] = f"data_name_{config['expname']}"
