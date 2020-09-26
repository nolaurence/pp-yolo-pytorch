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
# load config, merge-config, create
# Training Stats
