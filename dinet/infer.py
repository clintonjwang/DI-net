"""Entrypoint for inference"""
import sys
import torch
import numpy as np

from dinet.utils import args as args_module
from dinet.experiments.classify import test_inr_classifier
from dinet.experiments.segment import test_inr_segmenter

def main():
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    method_dict = {
        'classify': test_inr_classifier,
        'segment': test_inr_segmenter,
    }
    method_dict[args["network"]["task"]](args=args)

if __name__ == "__main__":
    main()
