import os, sys

# add python path of pp-yolo-pytorch to sys.path
parent_path = os.path.abspath('.')
if parent_path not in sys.path:
    sys.path.append(parent_path)

import time
import numpy as np
import random
import datetime
import six
from collections import deque

# modification
# paddle's profiler is await to find
import torch
from torch.optim import lr_scheduler  # decay step counter
from torch import optim  # ExponentialMovingAverage

from torch.cuda.amp import autocast, GradScaler  # mixed precision
from model.core.workspace import load_config, merge_config, create  # don't know about the function of create function
# from ppdet.data.reader import create_reader

from model.utils.cli import ArgsParser
from model.utils.check import check_gpu, check_version, check_config
# import model.utils.checkpoint as checkpoint
# Training Statsz

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def main():
    env = os.environ

    # “FLAGS” is the CLI attribute
    # don't know how it functions by now, maybe dist means 'distributed training'  --nolaurence
    FLAGS.dist = 'PADDLE_TRAINER_ID' in env \
                 and 'PADDLE_TRAINERS_NUM' in env \
                 and int(env['PADDLE_TRAINERS_NUM']) > 1
    num_trainers = int(env.get('PADDLE_TRAINERS_NUM', 1))  # trainers‘ number
    if FLAGS.dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    # internal test attr for paddle developer, will be moved in the future --nolaurence
    if FLAGS.enable_ce:
        random.seed(0)
        np.random.seed(0)

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)  # "opt" 's help text: set configuration options
                             # more information needs to be added
    check_config(cfg)
    check_gpu(cfg.use_gpu)  # check if set use_gpu=True in torch cpu version
    check_version()

    save_only = getattr(cfg)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Checkpoint path for resuming training.")
    parser.add_argument(
        "--fp16",
        action='store_true',
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--loss_scale",
        default=8.,
        type=float,
        help="Mixed precision training loss scale.")
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/scalar",
        help='VisualDL logging directory for scalar.')
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")

    #NOTE:args for profiler tools, used for benchmark
    parser.add_argument(
        '--is_profiler',
        type=int,
        default=0,
        help='The switch of profiler tools. (used for benchmark)')
    parser.add_argument(
        '--profiler_path',
        type=str,
        default="./detection.profiler",
        help='The profiler output file path. (used for benchmark)')
    FLAGS = parser.parse_args()
    main()
