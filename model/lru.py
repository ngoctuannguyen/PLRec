import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ExpertMLP(nn.Module):
    """A single expert: 2-layer MLP with SiLU activation."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))

# ============================================================================
# [OLD] Single-Gate MoE Category Predictor (commented out — replaced by Multi-Head MoE)
# ============================================================================
# class MoECategoryPredictor(nn.Module):
#     """Mixture of Experts for category prediction.
#     
#     Uses top-k sparse routing: only k experts are activated per input,
#     keeping compute cost manageable while increasing model capacity.
#     """
#     def __init__(self, input_dim, output_dim, num_experts=4,
#                  num_experts_per_token=2, hidden_dim=128):
#         super().__init__()
#         self.num_experts = num_experts
#         self.num_experts_per_token = num_experts_per_token
#
#         # Expert networks
#         self.experts = nn.ModuleList([
#             ExpertMLP(input_dim, hidden_dim, output_dim)
#             for _ in range(num_experts)
#         ])
#
#         # Gating network
#         self.gate = nn.Linear(input_dim, num_experts, bias=False)
#
#     def forward(self, x):
#         original_shape = x.shape[:-1]
#         input_dim = x.shape[-1]
#         x_flat = x.reshape(-1, input_dim)
#         T = x_flat.shape[0]
#         gate_logits = self.gate(x_flat)
#         if self.training:
#             noise = torch.randn_like(gate_logits) * 0.1
#             gate_logits = gate_logits + noise
#         top_k_logits, top_k_indices = torch.topk(
#             gate_logits, self.num_experts_per_token, dim=-1)
#         top_k_weights = F.softmax(top_k_logits, dim=-1)
#         expert_outputs = torch.stack(
#             [expert(x_flat) for expert in self.experts], dim=1)
#         selected_outputs = torch.gather(
#             expert_outputs, dim=1,
#             index=top_k_indices.unsqueeze(-1).expand(
#                 -1, -1, expert_outputs.shape[-1]))
#         output = (top_k_weights.unsqueeze(-1) * selected_outputs).sum(dim=1)
#         output = output.reshape(*original_shape, -1)
#         gate_probs = F.softmax(gate_logits, dim=-1)
#         expert_mask = torch.zeros(T, self.num_experts, device=x.device)
#         expert_mask.scatter_(1, top_k_indices, 1.0)
#         f = expert_mask.mean(dim=0)
#         P = gate_probs.mean(dim=0)
#         aux_loss = self.num_experts * (f * P).sum()
#         return output, aux_loss
# ============================================================================


class MultiHeadMoECategoryPredictor(nn.Module):
    """Multi-Head Mixture of Experts for category prediction.

    Splits the input into H heads, each with its own gate and expert pool.
    Each head routes independently, allowing diverse expert specialization:
      - Head 1 may focus on item collaborative features
      - Head 2 may focus on text/content features
      - etc.

    Final output is the concatenation of all head outputs, projected to
    the target category dimension.
    """
    def __init__(self, input_dim, output_dim, num_experts=4,
                 num_experts_per_token=2, hidden_dim=128, num_heads=2):
        super().__init__()
        assert input_dim % num_heads == 0, \
            f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.head_dim = input_dim // num_heads

        # Per-head expert pools: each head has its own set of experts
        # Expert input = head_dim, expert output = hidden_dim (intermediate)
        self.head_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertMLP(self.head_dim, hidden_dim, hidden_dim)
                for _ in range(num_experts)
            ])
            for _ in range(num_heads)
        ])

        # Per-head gating networks
        self.head_gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts, bias=False)
            for _ in range(num_heads)
        ])

        # Final projection: concat of H heads (each hidden_dim) → output_dim
        # self.output_proj = nn.Linear(num_heads * hidden_dim, output_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(num_heads * hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        self.layer_norm = nn.LayerNorm(num_heads * hidden_dim)

    def _route_single_head(self, x_head, x_full, experts, gate):
        """Route a single head's input through its experts.

        Args:
            x_head: (T, head_dim)
            x_full: (T, input_dim)
            experts: ModuleList of ExpertMLP
            gate: nn.Linear(input_dim, num_experts)

        Returns:
            head_output: (T, hidden_dim)
            aux_loss: scalar load balancing loss for this head
        """
        T = x_head.shape[0]

        # Gating based on full input representation
        gate_logits = gate(x_full)  # (T, num_experts)
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.1
            gate_logits = gate_logits + noise

        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(
            gate_logits, self.num_experts_per_token, dim=-1
        )  # (T, k)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (T, k)

        # Compute all expert outputs and gather selected ones
        expert_outputs = torch.stack(
            [expert(x_head) for expert in experts], dim=1
        )  # (T, num_experts, hidden_dim)

        selected_outputs = torch.gather(
            expert_outputs, dim=1,
            index=top_k_indices.unsqueeze(-1).expand(
                -1, -1, expert_outputs.shape[-1]
            ),
        )  # (T, k, hidden_dim)

        # Weighted combination
        head_output = (top_k_weights.unsqueeze(-1) * selected_outputs).sum(dim=1)  # (T, hidden_dim)

        # Load balancing loss for this head
        gate_probs = F.softmax(gate_logits, dim=-1)  # (T, num_experts)
        expert_mask = torch.zeros(T, self.num_experts, device=x_head.device)
        expert_mask.scatter_(1, top_k_indices, 1.0)
        f = expert_mask.mean(dim=0)   # fraction routed to each expert
        P = gate_probs.mean(dim=0)    # avg gate probability per expert
        aux_loss = self.num_experts * (f * P).sum()

        return head_output, aux_loss

    def forward(self, x):
        """Forward pass with multi-head expert gating.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            output: Category prediction logits, shape (..., output_dim)
            aux_loss: Total load balancing loss (sum across all heads)
        """
        original_shape = x.shape[:-1]
        input_dim = x.shape[-1]
        x_flat = x.reshape(-1, input_dim)  # (T, input_dim)

        # Split into heads: (T, input_dim) → H × (T, head_dim)
        x_heads = x_flat.chunk(self.num_heads, dim=-1)

        # Route each head independently
        head_outputs = []
        total_aux_loss = 0.0
        for i in range(self.num_heads):
            h_out, h_aux = self._route_single_head(
                x_heads[i], x_flat, self.head_experts[i], self.head_gates[i]
            )
            head_outputs.append(h_out)
            total_aux_loss = total_aux_loss + h_aux

        # Concatenate all head outputs and project to category space
        combined = torch.cat(head_outputs, dim=-1)  # (T, num_heads * hidden_dim)
        combined = self.layer_norm(combined)
        output = self.output_proj(combined)  # (T, output_dim)
        output = output.reshape(*original_shape, -1)

        # Average aux_loss across heads
        total_aux_loss = total_aux_loss / self.num_heads

        return output, total_aux_loss


class CrossModalityAttention(nn.Module):
    """
    Computes bidirectional cross-attention weights between ID and Text embeddings
    to fuse their representations dynamically.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2), 
            nn.Softmax(dim=-1)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.gate = nn.Parameter(torch.zeros(1))  # learnable residual gate, init at 0

    def forward(self, item_emb, txt_emb):
        combined = torch.cat([item_emb, txt_emb], dim=-1) 
        attn_weights = self.attn_net(combined) 
        
        w_id = attn_weights[..., 0:1] 
        w_txt = attn_weights[..., 1:2] 
        
        # Enriched representations via gated borrowing
        enriched_id = item_emb + w_txt * txt_emb
        enriched_txt = txt_emb + w_id * item_emb
        
        fused = torch.cat([enriched_id, enriched_txt], dim=-1) 
        return self.layer_norm(fused + torch.sigmoid(self.gate) * self.proj(fused))


class LRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = LRUEmbedding(self.args)
        self.model = LRUModel(self.args)
        self.truncated_normal_init()
        cat_emb = torch.load(f'./data/{args.dataset_code}/cat.pt').float()
        self.cat_embedding = nn.Embedding.from_pretrained(cat_emb).to(args.device)
        moe_num_experts = getattr(args, 'moe_num_experts', 4)
        moe_k = getattr(args, 'moe_num_experts_per_token', 2)
        moe_hidden = getattr(args, 'moe_hidden_dim', 128)
        moe_heads = getattr(args, 'moe_num_heads', 2)
        # [OLD] Single-gate MoE:
        # self.cat_predictor = MoECategoryPredictor(
        #     input_dim=2 * args.bert_hidden_units,
        #     output_dim=cat_emb.shape[-1],
        #     num_experts=moe_num_experts,
        #     num_experts_per_token=moe_k,
        #     hidden_dim=moe_hidden,
        # )
        # [NEW] Multi-Head MoE:
        self.cat_predictor = MultiHeadMoECategoryPredictor(
            input_dim=2 * args.bert_hidden_units,
            output_dim=cat_emb.shape[-1],
            num_experts=moe_num_experts,
            num_experts_per_token=moe_k,
            hidden_dim=moe_hidden,
            num_heads=moe_heads,
        )
        txt_emb = torch.load(f'./data/{args.dataset_code}/txt_embeddings.pt').float()
        self.txt_embedding = nn.Embedding.from_pretrained(txt_emb).to(args.device)
        self.txt_linear = nn.Linear(txt_emb.shape[-1], args.bert_hidden_units)
        self.modality_fusion = CrossModalityAttention(args.bert_hidden_units)

    def get_category_embedding(self):
        return self.cat_embedding

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def forward(self, x, labels=None):
        x, mask, pe = self.embedding(x)
        return self.model(x, self.embedding.token.weight, mask, labels=labels)

class LRUEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 1
        embed_size = args.bert_hidden_units
        
        self.token = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)
        self.positional_embedding = nn.Embedding(vocab_size, embed_size)

    def get_mask(self, x):
        return (x > 0)
    
    def forward(self, x):
        mask = self.get_mask(x)                   
        positional_ids = torch.cumsum(mask, dim=1)   
        positional_ids = positional_ids * mask
        pos_emb = self.positional_embedding(positional_ids)      
        x = self.token(x) + pos_emb
        return self.layer_norm(self.embed_dropout(x)), mask, pos_emb

class LRUModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_hidden_units
        # self.hidden_size =20
        layers = args.bert_num_blocks

        self.lru_blocks = nn.ModuleList([LRUBlock(self.args) for _ in range(layers)])
        self.bias = torch.nn.Parameter(torch.zeros(args.num_items + 1))

    def forward(self, x, embedding_weight, mask, labels=None):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]  # B x L x D (64)
        
        # prediction layer
        if self.args.dataset_code != 'xlong':
            scores = torch.matmul(x, embedding_weight.permute(1, 0)) + self.bias
            return scores, x
        else:
            assert labels is not None
            if self.training:
                num_samples = self.args.negative_sample_size  # 100
                samples = torch.randint(1, self.args.num_items+1, size=(*x.shape[:2], num_samples,))
                all_items = torch.cat([samples.to(labels.device), labels.unsqueeze(-1)], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b l i d -> b l i', x, sampled_embeddings) + self.bias[all_items]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_
            else:
                num_samples = self.args.xlong_negative_sample_size  # 10000
                samples = torch.randint(1, self.args.num_items+1, size=(x.shape[0], num_samples,))  # only one time step
                all_items = torch.cat([samples.to(labels.device), labels], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b i d -> b l i', x, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_.reshape(labels.shape)
            

class LRUBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_size = args.bert_hidden_units
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=args.bert_attn_dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size*4, dropout=args.bert_dropout)
    
    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x
    

class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above 
    
class SwiGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features * 2)
        self.linear2 = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        hidden_states = self.linear1(x)
        gate, activated = hidden_states.chunk(2, dim=-1)
        activated = F.silu(activated)
        output = self.linear2(gate * activated)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff * 2)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_proj = self.w_1(x)  # [B, L, d_ff*2]
        gate, act = x_proj.chunk(2, dim=-1)
        act = F.silu(act)
        x_ = self.dropout(gate * act)
        x_ = self.dropout(self.w_2(x_))
        return self.layer_norm(x_ + x)