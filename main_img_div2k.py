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

import importlib
import main


if __name__ == '__main__':
    config_fp = 'configs/img.py'

    config_fp = config_fp.replace('./', '').replace('.py', '').replace('/', '.')
    config = importlib.import_module(config_fp).config

    for i in range(801, 901):
        print(f'div2k {i}')

        config.workspace = f'log/img_div2k'
        config.path = f'./data/img/div2k/DIV2K_valid_HR/{str(i).zfill(4)}.png'

        config.pe_lc0_freq = [1e0, 1e3]
        config.pe_lc0_rbf_freq = [2**-3, 2**12]
        config.num_levels = 2
        config.level_dim = 2
        config.max_steps = 35000
        config.val_freq = 1.0
        config.val_first = False
        config.log_train = False
        config.log_img = False
        config.log_kernel_img = False
        config.log_sdf_slice = False
        config.save_pred = True
        config.save_pred_pt = False
        config.save_ckpt = False
        config.save_fn = str(i).zfill(4)
        config.exp_name = str(i).zfill(4)

        main.main(config)
