from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *
from .lru import LRUTrainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
from abc import *
from pathlib import Path


class NTNRecTrainer(LRUTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
    
    def CP(self, input, padding_idx=0):
        item_list = input
        nonzero_idx = torch.where(input != padding_idx)
        
        # NTNRecEmbedding doesn't use positional embeddings
        item_emb = self.model.embedding(item_list)[0]
        txt_emb = self.model.txt_embedding(item_list)
        txt_emb = self.model.txt_linear(txt_emb) 
        
        item_attribute_score = self.model.cat_linear(torch.cat([item_emb, txt_emb], dim=-1))
        item_attribute_target = self.model.cat_embedding(item_list)
        attr_loss = self.bce(item_attribute_score[nonzero_idx], item_attribute_target[nonzero_idx])
        return attr_loss
