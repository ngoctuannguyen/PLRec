import torch
import torch.nn as nn
import torch.nn.functional as F

class NTNRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = NTNRecEmbedding(self.args)
        self.model = NTNRecModel(self.args)
        
        cat_emb = torch.load(f'./data/{args.dataset_code}/cat.pt').float()
        self.cat_embedding = nn.Embedding.from_pretrained(cat_emb).to(args.device)
        self.cat_linear = nn.Linear(args.bert_hidden_units, cat_emb.shape[-1])
        
        txt_emb = torch.load(f'./data/{args.dataset_code}/txt_embeddings.pt').float()
        self.txt_embedding = nn.Embedding.from_pretrained(txt_emb).to(args.device)
        self.txt_linear = nn.Linear(txt_emb.shape[-1], args.bert_hidden_units)

    def get_category_embedding(self):
        return self.cat_embedding

    def forward(self, x, labels=None):
        x, mask = self.embedding(x)
        return self.model(x, self.embedding.token.weight, mask, labels=labels)

class NTNRecEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 1
        embed_size = args.bert_hidden_units
        
        self.token = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(args.bert_dropout)

    def get_mask(self, x):
        return (x > 0)
    
    def forward(self, x):
        mask = self.get_mask(x)                   
        x = self.token(x)
        return self.layer_norm(self.embed_dropout(x)), mask

class NTNRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_hidden_units
        
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=args.mc_num_gru_layers,
            batch_first=True,
            dropout=args.bert_dropout if args.mc_num_gru_layers > 1 else 0
        )
        
        self.ssc = SSCModule(
            hidden_size=self.hidden_size,
            chunk_size=args.mc_chunk_size,
            stride=args.mc_stride,
            top_k=args.mc_top_k
        )
        
        self.bias = nn.Parameter(torch.zeros(args.num_items + 1))

    def forward(self, x, embedding_weight, mask, labels=None):
        gru_out, _ = self.gru(x)
        
        augmented = self.ssc(gru_out)
        
        if self.args.dataset_code != 'xlong':
            scores = torch.matmul(augmented, embedding_weight.permute(1, 0)) + self.bias
            return scores, augmented
        else:
            assert labels is not None
            if self.training:
                num_samples = self.args.negative_sample_size
                samples = torch.randint(1, self.args.num_items+1, size=(*augmented.shape[:2], num_samples,), device=labels.device)
                all_items = torch.cat([samples, labels.unsqueeze(-1)], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b l i d -> b l i', augmented, sampled_embeddings) + self.bias[all_items]
                labels_ = torch.full_like(labels, num_samples)
                return scores, labels_
            else:
                num_samples = self.args.xlong_negative_sample_size
                samples = torch.randint(1, self.args.num_items+1, size=(augmented.shape[0], num_samples,), device=labels.device)
                all_items = torch.cat([samples, labels], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b i d -> b l i', augmented, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
                labels_ = torch.full_like(labels, num_samples)
                return scores, labels_

class SSCModule(nn.Module):
    def __init__(self, hidden_size, chunk_size, stride, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.stride = stride
        self.top_k = top_k
        
        self.w_u = nn.Linear(hidden_size, hidden_size)

    def forward(self, h_sequence):
        B, L, D = h_sequence.size()
        outputs = []
        
        max_mem = (L + self.stride - 1) // self.stride + 1
        if max_mem > 0:
            memory_tensor = torch.empty((B, max_mem, D), device=h_sequence.device, dtype=h_sequence.dtype)
            mean_pool_tensor = torch.empty((B, max_mem, D), device=h_sequence.device, dtype=h_sequence.dtype)
        num_mem = 0
        
        prev_end = 0
        
        # Xử lý Overlapping Chunking
        for c_start in range(0, L, self.stride):
            c_end = min(c_start + self.chunk_size, L)
            h_chunk = h_sequence[:, c_start:c_end, :]  # [B, S, D]
            S = h_chunk.size(1)
            
            if num_mem > 0:
                past_mems = memory_tensor[:, :num_mem, :].clone()  # [B, Num_Mem, D]
                past_means = mean_pool_tensor[:, :num_mem, :].clone()  # [B, Num_Mem, D]
                
                u_chunk = self.w_u(h_chunk)  # [B, S, D]
                past_scores = torch.bmm(u_chunk, past_means.transpose(1, 2)) / (self.hidden_size ** 0.5) # [B, S, Num_Mem]
                
                topk_k = min(self.top_k, num_mem)
                if num_mem > topk_k:
                    topk_vals, topk_idx = torch.topk(past_scores, topk_k, dim=-1)
                    mask = torch.full_like(past_scores, float('-inf'))
                    past_scores = mask.scatter_(-1, topk_idx, topk_vals)
                
                # Relevance score cho online chunk (Cumulative Mean)
                cum_sum = torch.cumsum(h_chunk, dim=1)
                lengths = torch.arange(1, S+1, device=h_chunk.device, dtype=h_chunk.dtype).view(1, S, 1)
                cum_mean = cum_sum / lengths
                online_scores = torch.sum(u_chunk * cum_mean, dim=-1, keepdim=True) / (self.hidden_size ** 0.5) # [B, S, 1]
                
                # Joint Softmax gating (Eq 17)
                joint_scores = torch.cat([online_scores, past_scores], dim=-1) # [B, S, 1 + Num_Mem]
                joint_probs = F.softmax(joint_scores, dim=-1) # [B, S, 1 + Num_Mem]
                
                online_prob = joint_probs[:, :, 0:1] # [B, S, 1]
                past_probs = joint_probs[:, :, 1:] # [B, S, Num_Mem]
                
                # Aggregation
                agg_past = torch.bmm(past_probs, past_mems) # [B, S, D]
                h_tilde_chunk = online_prob * h_chunk + agg_past
            else:
                h_tilde_chunk = h_chunk
                
            # Trích xuất các token mới để tránh trùng lặp ở output (ngăn Causal Leakage)
            if prev_end < c_end:
                start_idx_in_chunk = prev_end - c_start
                new_tokens = h_tilde_chunk[:, start_idx_in_chunk:, :]
                outputs.append(new_tokens)
                prev_end = c_end
            
            # Cập nhật Memory Buffers nếu đây là 1 chunk hoàn chỉnh
            if S == self.chunk_size:
                memory_tensor[:, num_mem, :] = h_tilde_chunk[:, -1, :].detach()
                mean_pool_tensor[:, num_mem, :] = h_chunk.mean(dim=1).detach()
                num_mem += 1
                
        return torch.cat(outputs, dim=1)
