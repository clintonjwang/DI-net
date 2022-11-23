"""
Entrypoint for training
"""
import sys
import torch, wandb
import numpy as np
from functools import partial

from dinet.utils import args as args_module
from dinet.experiments.sdf import train_nerf_to_sdf
from dinet.experiments.classify import train_classifier
from dinet.experiments.segment import train_segmenter

def main():
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])
    method_dict = {
        'classify': train_classifier,
        'sdf': train_nerf_to_sdf,
        'segment': train_segmenter,
    }
    method = method_dict[args["network"]["task"]]
    if args['sweep_id'] is not None:
        wandb.agent(args['sweep_id'], function=partial(method, args=args), count=1, project='dinet')
    else:
        if not args['no_wandb']:
            wandb.init(project="dinet", job_type="train", name=args["job_id"],
                config=wandb.helper.parse_config(args, exclude=['job_id']))
            args = args_module.get_wandb_config()
        method(args=args)

if __name__ == "__main__":
    main()
