# Modified from https://github.com/ashawkey/torch-ngp/blob/main/sdf/utils.py

import os
import glob
import tqdm
import random
import warnings
import tensorboardX
import imageio.v3 as iio
import math
import shutil

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from rich.console import Console
from torch_ema import ExponentialMovingAverage

import packaging
import util_metric
import util_network
import util_misc
import util_summary


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 model, # network 
                 hparams,
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 ema_decay=None, # if use EMA, set the decay
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 workspace='workspace', # workspace to save logs & ckpts
                 use_checkpoint="scratch", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=True, # whether to call scheduler.step() after every train step
                 ):
        self.name = name
        self.hparams = hparams
        self.mute = mute
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.train_metric_list = self.hparams.train_metric_list
        self.val_metric_list = self.hparams.val_metric_list
        self.log_train = self.hparams.log_train
        if not self.log_train: self.use_tensorboardX = False
        self.t_train = 0

        model.to(self.device)
        if self.world_size > 1:
            raise NotImplementedError
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        param_groups = util_network.get_param_groups(self.model)
        self.optims = util_network.configure_optimizers(param_groups, self.hparams.optims)
        self.lr_schs = util_network.configure_lr_schedulers(self.optims, self.hparams.lr_schs)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp32"} | {self.workspace}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            else:
                raise NotImplementedError

    def __del__(self):
        if hasattr(self, 'log_ptr') and self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def save_img(self, pred, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.png')
        self.log(f"==> Saving image to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img = (pred.clip(self.hparams.vmin, self.hparams.vmax).cpu().numpy() * 255).astype(np.uint8)
        iio.imwrite(save_path, img)
        self.log(f"==> Finished saving image.")

    def save_mesh(self, pred, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.obj')
        self.log(f"==> Saving mesh to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mesh = util_misc.extract_mesh(pred, 0, self.extract_mesh_cmin, self.extract_mesh_cmax)
        mesh.export(save_path)
        self.log(f"==> Finished saving mesh.")

    ### ------------------------------	

    def train_step(self, data):
        X = data["points"][0]  # [B, 2]
        y = data["gt"][0]  # [B, 3]
        point_idx = data["point_idx"][0]  # [B]
        
        pred, out_other = self.model(X, point_idx=point_idx)
        loss = self.criterion(pred, y)

        return pred, y, loss, out_other

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        X = data["points"][0]
        point_idx = data["point_idx"][0]
        pred = self.model(X, point_idx=point_idx)
        return pred        

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_steps):
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=max_steps, bar_format='{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            shutil.copy(self.hparams.config_fp, 
                os.path.join(self.workspace, "run", self.name, os.path.basename(self.hparams.config_fp)))
            
            if self.use_tensorboardX:
                self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

            if self.hparams.task == 'sdf':
                self.gt_mesh = train_loader.dataset.mesh
                self.extract_mesh_cmin = valid_loader.dataset.cmin.numpy()
                self.extract_mesh_cmax = valid_loader.dataset.cmax.numpy()
        
        self.max_steps = max_steps
        self.max_epochs = math.ceil(self.max_steps / len(train_loader))

        if self.hparams.val_first:
            self.pred_latest = self.evaluate_one_epoch(valid_loader)

        # Save initial pred
        if self.hparams.record_training:
            pred_latest = self.evaluate_one_epoch(valid_loader, val_metric_list=['psnr', 'mae', 'mse', 'ssim_ski', 'lpips_alex'], full=False)
            self.val_metrics_proc = {k: [v] for k, v in self.val_metrics.items()}
            save_dir = os.path.join(self.workspace, 'results', self.name)
            os.makedirs(save_dir, exist_ok=True)
            save_fn = os.path.join(save_dir, f'step_{str(self.global_step).zfill(6)}')
            iio.imwrite(f'{save_fn}.png', (pred_latest.clip(self.hparams.vmin, self.hparams.vmax).detach().cpu().numpy() * 255).astype(np.uint8))
            
        for epoch in range(self.epoch + 1, self.max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loader, pbar, valid_loader)

            if self.epoch % self.eval_interval == 0 or self.epoch == self.max_epochs:
                self.pred_latest = self.evaluate_one_epoch(valid_loader)

        if self.local_rank == 0:
            pbar.close()
            if self.use_tensorboardX:
                self.writer.close()

    def prepare_data(self, data):
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.to(self.device, non_blocking=True)
            else:
                raise NotImplementedError
        return data

    def train_one_epoch(self, loader, pbar, valid_loader):
        t = time.time()
        self.model.train()
        self.model.use_train_knn = True
        loader.dataset.shuffle_order(0)
        self.local_step = 0
        total_loss = 0
        self.t_train += time.time() - t
        metrics_str = ''

        for data in loader:
            t = time.time()
            if self.global_step == self.max_steps:
                break
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            for k, v in self.optims.items():
                v.zero_grad()

            preds, truths, loss, _ = self.train_step(data)
            loss.backward()

            for k, v in self.optims.items():
                v.step()

            if self.scheduler_update_every_step:
                for k, v in self.lr_schs.items():
                    v.step()

            loss_val = loss.item()
            total_loss += loss_val
            self.t_train += time.time() - t

            if self.local_rank == 0:
                loss_avg = total_loss/self.local_step

                # Save preds during training
                if self.hparams.record_training:
                    pred_latest = self.evaluate_one_epoch(valid_loader, val_metric_list=['psnr', 'mae', 'mse', 'ssim_ski', 'lpips_alex'], full=False)
                    for k, v in self.val_metrics.items():
                        self.val_metrics_proc[k].append(v)
                    if (self.global_step % 10 == 0 or self.global_step == self.max_steps):
                        save_dir = os.path.join(self.workspace, 'results', self.name)
                        save_fn = os.path.join(save_dir, f'step_{str(self.global_step).zfill(6)}')
                        iio.imwrite(f'{save_fn}.png', (pred_latest.clip(self.hparams.vmin, self.hparams.vmax).detach().cpu().numpy() * 255).astype(np.uint8))

                if self.log_train and ((self.global_step - 1) % 10 == 0 or self.local_step == 1 or self.global_step == self.max_steps):
                    self.train_metrics = self.compute_metrics(
                        preds.detach().clip(self.hparams.vmin, self.hparams.vmax), truths.detach(), 
                        self.train_metric_list)
                    if self.use_tensorboardX:
                        for k, v in self.optims.items():
                            self.writer.add_scalar(f"train/lr_{k}", v.param_groups[0]['lr'], self.global_step)
                        self.writer.add_scalar("train/loss", loss_val, self.global_step)
                        self.writer.add_scalar("train/loss_avg", loss_avg, self.global_step)
                        for k, v in self.train_metrics.items():
                            self.writer.add_scalar(f"train/{k}", v, self.global_step)
                    metrics_str = f"psnr={self.train_metrics['psnr']:.4f}"
                
                pbar.set_description(f"Epoch {self.epoch}/{self.max_epochs} training: loss={loss_val:.6f} ({loss_avg:.6f}), {metrics_str}, lr_dec={self.optims['dec'].param_groups[0]['lr']:.6f}")
                pbar.update(1)

        if not self.scheduler_update_every_step:
            raise NotImplementedError

    def compute_metrics(self, pred, gt, metrics):
        out = {}
        has_mesh_metrics = False
        for metric in metrics:
            if metric in ['mesh']:
                has_mesh_metrics = True
                break
        if has_mesh_metrics:
            pred_mesh = util_misc.extract_mesh(pred, 0, self.extract_mesh_cmin, self.extract_mesh_cmax)

        for metric in metrics:
            if metric == 'iou':
                out['iou'] = util_metric.iou(pred, gt, 0).item()
            elif metric == 'psnr':
                out['psnr'] = util_metric.psnr(pred, gt, vmin=self.hparams.vmin, vmax=self.hparams.vmax).item()
            elif metric == 'mae':
                out['mae'] = util_metric.mae(pred, gt).item()
            elif metric == 'mse':
                out['mse'] = ((pred - gt)**2).mean().item()
            elif metric == 'ssim':
                out['ssim'] = util_metric.ssim_func(pred, gt)
            elif metric == 'ssim_ski':
                out['ssim_ski'] = util_metric.ssim_ski_func(pred, gt, vmin=self.hparams.vmin, vmax=self.hparams.vmax)
            elif metric == 'ms-ssim':
                out['ms-ssim'] = util_metric.msssim(pred, gt)
            elif metric == 'lpips_alex':
                try:
                    out['lpips_alex'] = util_metric.rgb_lpips(pred, gt, net_name='alex', device=pred.device)
                except:
                    self.log("Failed to compute lpips_alex")
                    out['lpips_alex'] = None
            elif metric == 'lpips_vgg':
                try:
                    out['lpips_vgg'] = util_metric.rgb_lpips(pred, gt, net_name='vgg', device=pred.device)
                except:
                    self.log("Failed to compute lpips_vgg")
                    out['lpips_vgg'] = None
            elif metric == 'mesh':
                out.update(util_metric.mesh_metrics(pred_mesh, self.gt_mesh, 
                    self.hparams.metric_config['n_surface_samples'], self.hparams.metric_config['fscore_tau']))
        return out

    def evaluate_one_epoch(self, loader, val_metric_list=None, full=True):
        self.model.eval()
        if not self.hparams.train_val_same_points:
            self.model.use_train_knn = False
        else:
            self.model.use_train_knn = True
        total_loss = 0

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]                                               ')

        pred = torch.zeros([loader.dataset.n_point, self.model.out_dim])
        gt = torch.zeros([loader.dataset.n_point, self.model.out_dim])

        with torch.no_grad():
            self.local_step = 0
            for data in loader:    
                self.local_step += 1
                
                data = self.prepare_data(data)

                preds, truths, loss, out_other = self.eval_step(data)
                pred[data['point_idx']] = preds.cpu()
                gt[data['point_idx']] = truths.cpu()

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    pbar.set_description(f"Epoch {self.epoch}/{self.max_epochs} validation: loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                    pbar.update(loader.batch_size)

        loss_avg = total_loss / self.local_step
        pred.clip_(self.hparams.vmin, self.hparams.vmax)
        pred = pred.reshape(*loader.dataset.s_dims, -1)
        gt = gt.reshape(*loader.dataset.s_dims, -1)

        if self.local_rank == 0:
            if val_metric_list is None:
                val_metric_list = self.val_metric_list
            self.val_metrics = self.compute_metrics(pred, gt, val_metric_list)

            if full:
                metrics_str = f"psnr={self.val_metrics['psnr']:.4f}"
                if 'iou' in self.val_metrics:
                    metrics_str += f", iou={self.val_metrics['iou']:.6f}"
                pbar.set_description(f"Epoch {self.epoch}/{self.max_epochs} validation: loss={loss_avg:.6f}, {metrics_str}")
                pbar.close()

                if self.use_tensorboardX:
                    for k, v in self.val_metrics.items():
                        self.writer.add_scalar(f"val/{k}", v, self.global_step)

        if full:
            self.write_img_summary(loader.dataset, pred, gt)

        return pred

    def inference_step(self, points):
        return self.model(points, point_idx=None)

    def evaluate_points(self, points_dict):
        self.model.eval()
        self.model.use_train_knn = False
        metrics = {}

        for k in points_dict['points']:
            with torch.no_grad():
                pts = torch.tensor(points_dict['points'][k])
                gt = torch.tensor(points_dict['sdfs'][k])
                chunks = torch.split(pts, 2**18)
                pred = []
                for chunk_pts in tqdm.tqdm(chunks):
                    pred_i, _ = self.inference_step(chunk_pts.to(self.device))
                    pred.append(pred_i.cpu())
            pred = torch.cat(pred, dim=0)
            pred.clip_(self.hparams.vmin, self.hparams.vmax)
            gt.clip_(self.hparams.vmin, self.hparams.vmax)

            metrics[k] = {}
            metrics[k]['iou'] = util_metric.iou(pred, gt, 0).item()
            metrics[k]['psnr'] = util_metric.psnr(pred, gt, vmin=self.hparams.vmin, vmax=self.hparams.vmax).item()
            metrics[k]['mae'] = util_metric.mae(pred, gt).item()

        return metrics

    def write_img_summary(self, dataset, pred, gt):
        cmin, cmax = dataset.cmin.flip(-1).numpy(), dataset.cmax.flip(-1).numpy()  # x y ...
        if self.hparams.log_img:
            util_summary.write_image_summary_new(self.writer, self.global_step, 
                dataset.gt_cpu.movedim(-1, 0)[None], 
                dataset.gt_grad_cpu.movedim(-1, 0)[None], 
                pred.cpu().movedim(-1, 0)[None], 
                cmin, cmax, suffix='')
        if self.hparams.log_kernel_img and hasattr(self.model, 'get_kc'):
            for k in self.model.rbf_suffixes:
                kc = self.model.get_kc(k).detach().cpu().numpy()
                kw_sq = util_misc.ks_to_kw_sq(self.model.get_ks(k).detach().cpu().numpy(), self.model.rbf_type)
                util_summary.write_kernel_summary_2d(
                    self.writer, self.global_step, 
                    dataset.gt_cpu.permute(-1, 0, 1)[None], 
                    dataset.gt_grad_cpu.permute(-1, 0, 1)[None], 
                    pred.cpu().permute(-1, 0, 1)[None], 
                    kc, kw_sq, cmin, cmax, suffix='_' + k)
        if self.hparams.log_sdf_slice:
            util_summary.write_sdf_slice_summary(self.writer, self.global_step, 
                gt.cpu().movedim(-1, 0)[None], 
                pred.cpu().movedim(-1, 0)[None], 
                suffix='')

    def save_checkpoint(self, metrics):
        state = metrics
        state['epoch'] = self.epoch
        state['global_step'] = self.global_step
        state['model'] = self.model.state_dict()
        file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth"
        torch.save(state, file_path)
            
    def load_checkpoint(self, checkpoint=None):
        raise NotImplementedError
            