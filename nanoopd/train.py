
import os
import math
import time
import argparse
import socket as _socket
from statistics import mean
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from nanoopd.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanoopd.loss import compute_opd_loss
from nanoopd.rollout import (
    get_logprobs,
    generate_rollouts_remote,
    remote_vllm_init_weight_transfer,
    sync_weights_to_vllm_inplace,
    prepare_batch,
    wait_for_rollout_worker,
)

