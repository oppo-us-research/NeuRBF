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
os.environ["OMP_NUM_THREADS"] = "256"

import sys
import importlib
import time
import glob
import math
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # enable reading large image
import GPUtil
import trimesh

import numpy as np
import torch

import img_sdf.utils as utils
import util_misc
import util_network
import util_metric
import pprint
import pickle


def main(opt):
    # Select devices
    opt.device = util_misc.select_devices('1#', force_reselect=True, excludeID=[])[0]
    print(f'GPU to be used: {opt.device}')
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opt.device}"
    temp = torch.ones([100, 100], device=0)
    del temp

    if opt.seed is not None:
        utils.seed_everything(opt.seed)

    import img_sdf.network as network

    # Process experiment name
    exp_list = glob.glob(f'{opt.workspace}/run/{opt.exp_name}-v*'.replace('[', '[[]'))
    if len(exp_list) == 0:
        vid = 0
    else:
        vid = max([int(v.split(f'{opt.exp_name}-v')[-1]) for v in exp_list]) + 1
    opt.exp_name += f'-v{vid}'
    os.makedirs(os.path.join(opt.workspace, "run", opt.exp_name), exist_ok=True)        
    print(f"[INFO] exp_name: {opt.exp_name}")

    # Load data
    if opt.task == 'image':
        gt = Image.open(opt.path).convert('RGB')
        if opt.path.endswith('.jpg') or opt.path.endswith('.png') or opt.path.endswith('.tif'):
            gt = np.uint8(gt)  # [h w c]
        else:
            raise NotImplementedError
        gt = torch.tensor(gt)  # [h w c]
        print(f"[INFO] image: {gt.shape} {opt.path}")

        s_dims = gt.shape[:-1]
        n_train_point = torch.tensor(s_dims).prod()
        if n_train_point > torch.iinfo(torch.int).max: raise NotImplementedError
        if n_train_point >= 50000*20000:
            opt.kc_init_config['0']['points_sampling'] = 2
            if 'lpips_alex' in opt.val_metric_list: opt.val_metric_list.remove('lpips_alex')
        if n_train_point >= 3000*3000:
            if 'lpips_vgg' in opt.val_metric_list: opt.val_metric_list.remove('lpips_vgg')

        if opt.cfactor is None:
            opt.cmax = [1, 1]  # ... y x
        else:
            opt.cmax = (torch.tensor(gt.shape[:2]) / opt.cfactor).tolist()  # ... y x
        opt.cmin = [-i for i in opt.cmax]  # ... y x
        desired_resolution = max(gt.shape[0:2]) // 2
        in_dim = 2
        out_dim = 3
        opt.vmin = 0
        opt.vmax = 1
    elif opt.task == 'sdf':
        gt_mesh = trimesh.load(opt.path, force='mesh')
        print(f"[INFO] mesh: {gt_mesh.vertices.shape} {gt_mesh.faces.shape} {opt.path}")
        opt.cmax = [1, 1, 1]  # ... y x
        opt.cmin = [-i for i in opt.cmax]  # ... y x
        s_dims = [opt.val_resolution] * 3
        n_train_point = opt.train_num_samples * opt.train_epoch_size
        desired_resolution=opt.desired_resolution
        in_dim = 3
        out_dim = 1
        opt.vmin = -(2**2*3)**0.5
        opt.vmax = (2**2*3)**0.5
    else:
        raise NotImplementedError
    
    print('Prepare dataset and dataloader')
    if opt.ds_device == 'auto':
        device = 'cpu' if n_train_point > int(5e8) else 0
    elif opt.ds_device == 'cpu':
        device = 'cpu'
    else:
        device = int(opt.ds_device)
    if opt.task == 'image':
        from img_sdf.provider import IMGDataset
        train_dataset = IMGDataset(gt, opt.cmin, opt.cmax, s_dims, num_samples=opt.train_num_samples, ns_per_block=1, 
            shuffle_mode=opt.train_shuffle_mode, device=device)
        valid_dataset = train_dataset
        opt.train_val_same_points = True
        train_pin_memory = False
        valid_pin_memory = False
    elif opt.task == 'sdf':
        from img_sdf.provider import SDFDataset
        train_dataset = SDFDataset(gt_mesh, opt.cmin, opt.cmax, s_dims, num_samples=opt.train_num_samples, 
            size=opt.train_epoch_size, presample=opt.train_presample, shuffle_mode=opt.train_shuffle_mode, 
            clip_sdf=opt.clip_sdf, mesh_fp=opt.path, device=device)
        valid_dataset = SDFDataset(gt_mesh, opt.cmin, opt.cmax, s_dims, num_samples=2**18, is_grid=True,
            clip_sdf=opt.clip_sdf, mesh_fp=opt.path, device=device)
        opt.train_val_same_points = False
        train_pin_memory = True if not train_dataset.presample else False
        valid_pin_memory = True if not valid_dataset.presample else False
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, 
        num_workers=0, pin_memory=train_pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, 
        num_workers=0, pin_memory=valid_pin_memory)

    print('Init model')
    if opt.arch == 'ngp':
        t = time.time()
        model = network.IMGNetwork(opt.cmin, opt.cmax, encoding=opt.encoding, num_layers=opt.num_layers, 
            hidden_dim=opt.hidden_dim, in_dim=in_dim, out_dim=out_dim,
            num_levels=opt.num_levels, level_dim=opt.level_dim, base_resolution=opt.base_resolution,
            log2_hashmap_size=opt.log2_hashmap_size, desired_resolution=desired_resolution, 
            act=opt.act, lc_act=opt.lc_act,
            lc_init=opt.lc_init, lca_init=opt.lca_init, w_init=opt.w_init, b_init=opt.b_init, a_init=opt.a_init,
            pe_freqs=opt.pe_freqs, levels_omit=opt.levels_omit)
        t_init_model = time.time() - t
        t_init_rbf = 0
        t_knn = 0
    else:
        t = time.time()
        net = eval(f'network.{opt.arch}')
        model = net(opt.cmin, opt.cmax, s_dims, in_dim=in_dim, out_dim=out_dim,
            num_layers=opt.num_layers, hidden_dim=opt.hidden_dim, n_hidden_fl=opt.n_hidden_fl, 
            num_levels_ref=opt.num_levels_ref, level_dim_ref=opt.level_dim_ref, 
            base_resolution_ref=opt.base_resolution_ref, log2_hashmap_size_ref=opt.log2_hashmap_size_ref, 
            num_levels=opt.num_levels, level_dim=opt.level_dim, base_resolution=opt.base_resolution,
            log2_hashmap_size=opt.log2_hashmap_size, desired_resolution=desired_resolution, 
            rbf_type=opt.rbf_type, n_kernel=opt.n_kernel, point_nn_kernel=opt.point_nn_kernel, ks_alpha=opt.ks_alpha, 
            lc_init=opt.lc_init, lcb_init=opt.lcb_init, 
            w_init=opt.w_init, b_init=opt.b_init, a_init=opt.a_init,
            sparse_embd_grad=False, act=opt.act, lc_act=opt.lc_act, rbf_suffixes=opt.rbf_suffixes, 
            kc_init_config=opt.kc_init_config, rbf_lc0_normalize=opt.rbf_lc0_normalize, 
            pe_freqs=opt.pe_freqs, pe_lc0_freq=opt.pe_lc0_freq, pe_hg0_freq=opt.pe_hg0_freq,
            pe_lc0_rbf_freq=opt.pe_lc0_rbf_freq, pe_lc0_rbf_keep=opt.pe_lc0_rbf_keep)
        t_init_model = time.time() - t
        t = time.time()
        util_network.init_rbf_params(model, train_dataset, opt.kc_init_config, opt.kw_init_config, device=0)
        t_init_rbf = time.time() - t
        t = time.time()
        if hasattr(train_dataset, 'points'):
            model.update_point_kernel_idx(train_dataset.points.view(-1, train_dataset.points.shape[-1]), device=device)
        t_knn = time.time() - t
    util_network.fix_params(model, opt.fix_params)
    model.count_params()

    if opt.task == 'image':
        train_dataset.prepare()
    elif opt.task == 'sdf':
        train_dataset.prepare()
        valid_dataset.prepare()

    if opt.task == 'image':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif opt.task == 'sdf':
        from thirdparty.torch_ngp.loss import mape_loss
        criterion = mape_loss

    max_steps = opt.max_steps
    max_epochs = math.ceil(max_steps / len(train_loader))
    eval_interval = math.ceil(max_epochs * opt.val_freq)

    print('Set up trainer')
    trainer = utils.Trainer(opt.exp_name, model, hparams=opt, workspace=opt.workspace, criterion=criterion,
        ema_decay=opt.ema_decay, fp16=opt.fp16, use_checkpoint='scratch', eval_interval=eval_interval, local_rank=0)

    # Train
    trainer.train(train_loader, valid_loader, max_steps=max_steps)
    t_train = trainer.t_train
    metrics = {'grid': trainer.val_metrics}

    # Eval
    if opt.task == 'sdf':
        eval_data = torch.load(opt.path.rsplit('.', 1)[0] + '_eval_points.pt')
        metrics.update(trainer.evaluate_points(eval_data))

    # Print stats
    stats = {}
    stats['time'] = {'t_init_model': t_init_model, 't_init_rbf': t_init_rbf, 't_knn': t_knn, 't_train': t_train, 
                        't_total': t_init_model + t_init_rbf + t_knn + t_train}
    stats['misc'] = {'n_epoch': trainer.epoch, 'n_iter': trainer.global_step, 'batch_size': opt.train_num_samples}
    stats['n_param'] = {f'n_param_{k}': v.tolist() for k, v in model.count_params().items()}
    stats['metrics'] = metrics
    pprint.pprint(stats, sort_dicts=False)

    # Write stats to file
    os.makedirs(os.path.join(opt.workspace, "results"), exist_ok=True)        
    save_fn = os.path.join(opt.workspace, 'results', opt.exp_name)
    with open(f'{save_fn}.txt', 'w') as f:
        pprint.pprint(stats, f, sort_dicts=False)

    if opt.save_pred:
        if opt.task == 'image':
            trainer.save_img(trainer.pred_latest, f'{save_fn}.png')
        elif opt.task == 'sdf':
            trainer.save_mesh(trainer.pred_latest, f'{save_fn}.ply')

    if opt.save_pred_pt:
        torch.save(trainer.pred_latest.detach().cpu(), f'{save_fn}.pt')
    
    if opt.save_ckpt:
        trainer.save_checkpoint(metrics)

    if hasattr(opt, 'save_kcs0') and opt.save_kcs0:
        torch.save({'kc0': trainer.model.kc0.weight.detach().cpu(), 'ks0': trainer.model.ks0.weight.detach().cpu()}, 
                    f'{save_fn}_kcs0.pt')

    # Save process data
    if opt.record_training:
        with open(f'{save_fn}.pkl', 'wb') as fp:
            pickle.dump(trainer.val_metrics_proc, fp)


if __name__ == '__main__':
    config_fp = sys.argv[2].replace('./', '').replace('.py', '').replace('/', '.')
    opt = importlib.import_module(config_fp).config
    main(opt)
