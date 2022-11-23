"""
Argument parsing
"""
import argparse
import os
import shutil
import wandb
import yaml
from glob import glob
from dinet import CONFIG_DIR

osp = os.path

def parse_args(args):
    """Command-line args for train.sh, infer.sh and sweep.sh"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name')
    parser.add_argument('-j', '--job_id', default="manual")
    parser.add_argument('-t', '--target_job', default=None)
    parser.add_argument('-i', '--sweep_id', default=None)
    parser.add_argument('-w', '--no_wandb', action='store_true')
    cmd_args = parser.parse_args(args)

    config_name = cmd_args.job_id if cmd_args.config_name is None else cmd_args.config_name
    configs = glob(osp.join(CONFIG_DIR, "*", config_name+".yaml"))
    if len(configs) == 0:
        raise ValueError(f"Config {config_name} not found in {CONFIG_DIR}")
    main_config_path = configs[0]

    args = args_from_file(main_config_path, cmd_args)
    paths = args["paths"]
    for subdir in ("weights", "imgs"):
        shutil.rmtree(osp.join(paths["job output dir"], subdir), ignore_errors=True)
    for subdir in ("weights", "imgs"):
        os.makedirs(osp.join(paths["job output dir"], subdir))
    yaml.safe_dump(args, open(osp.join(paths["job output dir"], "config.yaml"), 'w'))
    return args

def parse_args_fitting(args):
    """Command-line args for fit_inr.sh"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-j', '--job_id', default="manual")
    parser.add_argument('-s', '--start_ix', default=0, type=int)
    cmd_args = parser.parse_args(args)

    args = yaml.load(open(CONFIG_DIR+'/fit_config.yaml', 'r'))[cmd_args.dataset]
    args['data loading']['dataset'] = cmd_args.dataset
    for param in ["job_id", 'start_ix']:
        args[param] = getattr(cmd_args, param)
    yaml.safe_dump(args, open(osp.join(args["paths"]["job output dir"], "config.yaml"), 'w'))
    return args

def get_wandb_config():
    wandb_sweep_dict = {
        'learning_rate': ['optimizer', 'learning rate'],
        'batch_size': ['data loading', 'batch size'],
        'sample_points': ['data loading', 'sample points'],
        'discretization_type': ['data loading', 'discretization type'],
        'augment': ['data loading', 'augment'],
        'weight_decay': ['optimizer', 'weight decay'],
        'optimizer_type': ['optimizer', 'type'],
        
        'kernel_size_0': ['network', 'conv', 'k0'],
        'kernel_size_1': ['network', 'conv', 'k1'],
        'kernel_size_2': ['network', 'conv', 'k2'],
        'kernel_size_3': ['network', 'conv', 'k3'],
        'posenc_order': ['network', 'conv', 'posenc_order'],
        'pe_scale': ['network', 'conv', 'pe_scale'],
        'conv_mlp_type': ['network', 'conv', 'mlp_type'],
        'conv_N_bins': ['network', 'conv', 'N_bins'],
        'conv_N_ch': ['network', 'conv', 'mid_ch'],
    }
    if wandb.config['sweep_id'] is not None:
        for k,subkeys in wandb_sweep_dict.items():
            if k in wandb.config:
                d = wandb.config
                for subk in subkeys[:-1]:
                    d = d[subk]
                d[subkeys[-1]] = wandb.config[k]
    wandb.config.persist()
    return dict(wandb.config.items())

def infer_missing_args(args):
    """Convert str to float, etc."""
    paths = args["paths"]
    paths["slurm output dir"] = osp.expanduser(paths["slurm output dir"])
    if args["job_id"].startswith("lg_") or args["job_id"].startswith("A6"):
        args["data loading"]["batch size"] *= 6
        args["optimizer"]["epochs"] //= 6
    paths["job output dir"] = osp.join(paths["slurm output dir"], args["job_id"])
    paths["weights dir"] = osp.join(paths["job output dir"], "weights")
    for k in args["optimizer"]:
        if "learning rate" in k:
            args["optimizer"][k] = float(args["optimizer"][k])
    args["optimizer"]["weight decay"] = float(args["optimizer"]["weight decay"])

def merge_args(parent_args, child_args):
    """Merge parent config args into child configs."""
    if "_overwrite_" in child_args.keys():
        return child_args
    for k,parent_v in parent_args.items():
        if k not in child_args.keys():
            child_args[k] = parent_v
        else:
            if isinstance(child_args[k], dict) and isinstance(parent_v, dict):
                child_args[k] = merge_args(parent_v, child_args[k])
    return child_args


def args_from_file(path, cmd_args=None):
    """Create args dict from yaml."""
    if osp.exists(path):
        args = yaml.safe_load(open(path, 'r'))
    else:
        raise ValueError(f"bad config_name {cmd_args.config_name}")

    if cmd_args is not None:
        for param in ["job_id", "config_name", 'target_job', 'sweep_id', 'no_wandb']:
            args[param] = getattr(cmd_args, param)

    while "parent" in args:
        if isinstance(args["parent"], str):
            config_path = glob(osp.join(CONFIG_DIR, "*", args.pop("parent")+".yaml"))[0]
            args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
        else:
            parents = args.pop("parent")
            for p in parents:
                config_path = glob(osp.join(CONFIG_DIR, "*", p+".yaml"))[0]
                args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
            if "parent" in args:
                raise NotImplementedError("need to handle case of multiple parents each with other parents")

    config_path = osp.join(CONFIG_DIR, "default.yaml")
    args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
    infer_missing_args(args)
    return args
