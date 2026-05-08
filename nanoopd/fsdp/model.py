from typing import Dict, Optional
from omegaconf import DictConfig
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoConfig
