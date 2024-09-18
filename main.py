import argparse
import torch
import numpy as np
import random
from resnet_trainer import Resnet_Trainer


def main(args):
    trainer = Resnet_Trainer(args)
    if args.visualize:
        trainer.visualize_distribution()
    else:
        trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize",    action='store_true',           help="only load model and visualize results")
    parser.add_argument("--save-path",    type=str,   default="",        help="folder to save model")
    parser.add_argument("--log-dir",      type=str,   default="",        help="folder to log results")
    parser.add_argument("--epochs",       type=int,   default=90,        help="the total epoch to train model")
    parser.add_argument("--seed",         type=int,   default=3072,      help="fix seed")
    parser.add_argument("--data",         type=str,   default='combine', help="load data")
    parser.add_argument("--sample-rate",  type=float, default=0,         help="the sample rate of data")
    parser.add_argument("--save-data", action='store_true', help="save data")

    args = parser.parse_args()

    params = {
        "dim_z" : 64,
        "num_rb":  2,
        "flg_bn": True
    }
    args.params = params

    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.deterministic = True
    main(args)