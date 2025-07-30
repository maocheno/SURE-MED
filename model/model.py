import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import AutoImageProcessor
from models.chexbert import CheXbert
import numpy as np
import pandas as pd
from models.metrics import compute_mlc
from transformers import get_cosine_schedule_with_warmup
import ipdb
