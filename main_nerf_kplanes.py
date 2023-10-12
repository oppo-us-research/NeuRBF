# Modified from https://github.com/sarafridov/K-Planes/blob/main/plenoxels/main.py

import os

import time
import argparse
import importlib.util
import logging
import pprint
import sys
from typing import List, Dict, Any
import tempfile
import glob

import numpy as np

import util_misc


def get_freer_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fname = os.path.join(tmpdir, "tmp")
        os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >"{tmp_fname}"')
        if os.path.isfile(tmp_fname):
            memory_available = [int(x.split()[2]) for x in open(tmp_fname, 'r').readlines()]
            if len(memory_available) > 0:
                return np.argmax(memory_available)
    return None  # The grep doesn't work with all GPUs. If it fails we ignore it.

gpu = util_misc.select_devices('1#', force_reselect=True, excludeID=[])[0]
if gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"CUDA_VISIBLE_DEVICES set to {gpu}")
else:
    print(f"Did not set GPU.")


import torch
import torch.utils.data
from thirdparty.kplanes.runners import video_trainer
from thirdparty.kplanes.runners import phototourism_trainer
from thirdparty.kplanes.runners import static_trainer
from thirdparty.kplanes.utils.create_rendering import render_to_path, decompose_space_time
from thirdparty.kplanes.utils.parse_args import parse_optfloat


temp = torch.ones([100, 100], device=0)
del temp


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers,
                        force=True)


def load_data(model_type: str, data_downsample, data_dirs, validate_only: bool, render_only: bool, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)

    if model_type == "video":
        return video_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)
    elif model_type == "phototourism":
        return phototourism_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs
        )
    else:
        return static_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)


def init_trainer(model_type: str, **kwargs):
    if model_type == "video":
        from thirdparty.kplanes.runners import video_trainer
        return video_trainer.VideoTrainer(**kwargs)
    elif model_type == "phototourism":
        from thirdparty.kplanes.runners import phototourism_trainer
        return phototourism_trainer.PhototourismTrainer(**kwargs)
    else:
        from thirdparty.kplanes.runners import static_trainer
        return static_trainer.StaticTrainer(**kwargs)


def save_config(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
        for key in config.keys():
            f.write("%s\t%s\n" % (key, config[key]))


def main(args, init_data=None):
    setup_logging()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config), args.config)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config

    # Process overrides from argparse into config
    if args.data_name is not None:
        config['data_dirs'] = [config['data_dirs'][0].rsplit('/', 1)[0] + '/' + args.data_name]
    overrides: List[str] = args.override
    overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}
    config.update(overrides_dict)

    # Replace placeholders with actual value
    config['data_fn'] = config['data_dirs'][0].split('/')[-1]
    config['expname'] = config['expname'].replace('data_name', config['data_fn'])
    if 'rbf_config' in config:
        if 'init_data_fp' in config['rbf_config']:
            config['rbf_config']['init_data_fp'] = config['rbf_config']['init_data_fp'].replace('data_name', config['data_fn'])

    # Add version id
    exp_list = glob.glob(f'{config["logdir"]}/{config["expname"]}-v*'.replace('[', '[[]'))
    if len(exp_list) == 0:
        vid = 0
    else:
        vid = max([int(v.split(f'{config["expname"]}-v')[-1]) for v in exp_list]) + 1
    config["expname"] += f'-v{vid}'

    # If will load trained model
    if args.log_dir is not None:
        config['expname'] = args.log_dir.split('/')[-1]
        if 'rbf_config' in config:
            config['rbf_config']['init_rbf'] = False

    if "keyframes" in config:
        model_type = "video"
    elif "appearance_embedding_dim" in config:
        model_type = "phototourism"
    else:
        model_type = "static"
    validate_only = args.validate_only
    render_only = args.render_only
    spacetime_only = args.spacetime_only
    if validate_only and render_only:
        raise ValueError("render_only and validate_only are mutually exclusive.")
    if render_only and spacetime_only:
        raise ValueError("render_only and spacetime_only are mutually exclusive.")
    if validate_only and spacetime_only:
        raise ValueError("validate_only and spacetime_only are mutually exclusive.")

    pprint.pprint(config)
    print(config["expname"])

    if validate_only or render_only:
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
    else:
        save_config(config)

    data = load_data(model_type, validate_only=validate_only, render_only=render_only or spacetime_only, **config)
    config.update(data)
    config['init_data'] = init_data
    trainer = init_trainer(model_type, **config)
    if args.log_dir is not None:
        checkpoint_path = os.path.join(args.log_dir, "model.pth")
        training_needed = not (validate_only or render_only or spacetime_only)
        trainer.load_model(torch.load(checkpoint_path), training_needed=training_needed)

    if validate_only:
        trainer.validate()
    elif render_only:
        render_to_path(trainer, extra_name=config['expname'].split('-')[0])
    elif spacetime_only:
        decompose_space_time(trainer, extra_name="")
    else:
        t_train = time.time()
        init_data = trainer.train()
        t_train = time.time() - t_train

        stats_t = {'t_init': trainer.model.t_init, 't_train': t_train, 't_total': trainer.model.t_init + t_train}
        stats_misc = {'n_iter': config["num_steps"], 'batch_size': config["batch_size"], 'n_param': trainer.model.n_params}
        if hasattr(trainer, 'val_metrics'):
            stats_metric = trainer.val_metrics[0]
        else:
            stats_metric = {}
        print(stats_t)
        print(stats_misc)
        print(stats_metric)
        
        save_fn = os.path.join(config['logdir'], config['expname'], 'stats')
        with open(f'{save_fn}.txt', 'w') as f:
            pprint.pprint(stats_t, f, sort_dicts=False)
            pprint.pprint(stats_misc, f, sort_dicts=False)
            pprint.pprint(stats_metric, f, sort_dicts=False)

        return init_data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="")
    p.add_argument('--data_name', type=str, default=None)
    p.add_argument('--render-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--spacetime-only', action='store_true')
    p.add_argument('--config_init', type=str, default=None)
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('override', nargs=argparse.REMAINDER)
    args = p.parse_args()

    init_data = None
    if args.config_init is not None:
        args_config = args.config
        args.config = args.config_init
        print('Get init data...')
        init_data = main(args)
        torch.cuda.empty_cache()
        args.config = args_config

    main(args, init_data)
