from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import os
import numpy as np
from abc import *
from pathlib import Path


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LRUTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        # self.bce = nn.BCEWithLogitsLoss()
        self.bce = FocalLoss(alpha=0.25, gamma=2.0)
        
        # --- Dynamic Load Balancing Vars ---
        self.dynamic_lb_weight = getattr(self.args, 'moe_load_balance_weight', 0.01)
        self.lb_moving_avg = 1.0  # Mức lý tưởng là 1.0 (cân bằng hoàn hảo)
        self.lb_alpha = 0.99      # Hệ số làm mượt (smoothing factor)
        self.batch_count = 0

    def calculate_loss(self, batch):
        seqs, labels = batch
        
        if self.args.dataset_code != 'xlong':
            logits, hidden_items = self.model(seqs)
            if labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1):
                # Validation mode: only the last item is predicted
                logits = logits[:, -1, :]
                labels = labels.view(-1)
            else:
                # Training mode: predict all items in sequence
                logits = logits.reshape(-1, logits.size(-1))
                labels = labels.reshape(-1)
            
            # Tính toán Category Prediction và Aux Loss
            attr_loss, moe_aux_loss = self.CP(seqs)
            
            # --- Auto-Adjust Dynamic Load Balancing ---
            with torch.no_grad():
                aux_val = moe_aux_loss.item()
                if getattr(self.model, 'training', True):
                    self.lb_moving_avg = self.lb_alpha * self.lb_moving_avg + (1 - self.lb_alpha) * aux_val
                    
                    # Nếu sụp đổ (> 2.0 cho 4 experts), tăng hình phạt thêm 5%
                    if self.lb_moving_avg > 2.0:
                        self.dynamic_lb_weight = min(0.2, self.dynamic_lb_weight * 1.05)
                    # Nếu an toàn (< 1.5), giảm hình phạt về mức mặc định
                    elif self.lb_moving_avg < 1.5:
                        default_w = getattr(self.args, 'moe_load_balance_weight', 0.01)
                        self.dynamic_lb_weight = max(default_w, self.dynamic_lb_weight * 0.95)
                        
                    # Logging mỗi 50 batch
                    self.batch_count += 1
                    if self.batch_count % 50 == 0:
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "MoE/Aux_Loss": aux_val,
                                "MoE/LB_Moving_Avg": self.lb_moving_avg,
                                "MoE/Dynamic_Weight": self.dynamic_lb_weight
                            })
                        else:
                            self.logger_service.writer.add_scalar("MoE/Aux_Loss", aux_val, self.batch_count)
                            self.logger_service.writer.add_scalar("MoE/LB_Moving_Avg", self.lb_moving_avg, self.batch_count)
                            self.logger_service.writer.add_scalar("MoE/Dynamic_Weight", self.dynamic_lb_weight, self.batch_count)

            loss = self.ce(logits, labels) + \
                  self.args.CP_loss_weight * (attr_loss + self.dynamic_lb_weight * moe_aux_loss)
        else:
            logits, labels_ = self.model(seqs, labels=labels)
            logits = logits.reshape(-1, logits.size(-1))
            labels_[labels==0] = 0
            labels_ = labels_.view(-1)
            loss = self.ce(logits, labels_)
        return loss 
    
    def CP(self, input, padding_idx=0):
        item_list = input
        nonzero_idx = torch.where(input != padding_idx)
        emb_output = self.model.embedding(item_list)
        item_emb, pos_emb = emb_output[0], emb_output[2]
        txt_emb = self.model.txt_embedding(item_list)
        txt_emb = self.model.txt_linear(txt_emb) + pos_emb
        
        # Cross-modality attention fusion instead of simple concat
        moe_input = self.model.modality_fusion(item_emb, txt_emb)
        
        item_attribute_score, moe_aux_loss = self.model.cat_predictor(moe_input)
        item_attribute_target = self.model.cat_embedding(item_list)
        attr_loss = self.bce(item_attribute_score[nonzero_idx], item_attribute_target[nonzero_idx])
        return attr_loss, moe_aux_loss
    
    def IDCL(self, seqs, logits):
        # logits: [batch_size, seq_len, embed_dim]
        # pos_items_emb: [batch_size, seq_len, embed_dim]
        logits = F.normalize(logits, dim=-1)
        pos_items_emb, _ = self.model.embedding(seqs)
        pos_items_emb = F.normalize(pos_items_emb, dim=-1)

        batch_size, seq_len, _ = logits.shape

        # Positive logits: dot product at same position
        pos_logits = (logits * pos_items_emb).sum(dim=-1) / 0.2  # [batch_size, seq_len]
        pos_logits = torch.exp(pos_logits)

        # Negative logits: compare each sequence with all others in batch at same position
        # [batch_size, seq_len, embed_dim] x [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, batch_size]
        neg_logits = torch.einsum('bld,mld->blm', logits, pos_items_emb) / 0.2  # [batch_size, seq_len, batch_size]

        # Mask out self-comparisons (diagonal in batch)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=seqs.device)  # [batch_size, batch_size]
        mask = mask.unsqueeze(1).expand(batch_size, seq_len, batch_size)     # [batch_size, seq_len, batch_size]
        neg_logits = torch.where(mask, neg_logits, torch.tensor(0.0, device=neg_logits.device))

        neg_logits = torch.exp(neg_logits).sum(dim=-1)  # [batch_size, seq_len]

        # Final loss
        loss = -torch.log(pos_logits / (neg_logits + 1e-8))  # avoid division by zero
        return loss.mean()

    def calculate_metrics(self, batch):
        seqs, labels = batch
        
        if self.args.dataset_code != 'xlong':
            scores = self.model(seqs)[0][:, -1, :]
            B, L = seqs.shape
            for i in range(L):
                scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9  # padding
        else:
            scores, labels = self.model(seqs, labels=labels)
            scores = scores[:, -1, :]
        
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        return metrics