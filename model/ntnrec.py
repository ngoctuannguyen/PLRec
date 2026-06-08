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

class GRULocalEncoder(nn.Module):
    def __init__(self, gru):
        super().__init__()
        self.gru = gru
    def forward(self, x, mask=None):
        out, _ = self.gru(x)
        return out

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
        
        self.ssc = SSCWrapper(
            local_encoder=GRULocalEncoder(self.gru),
            hidden_size=self.hidden_size,
            chunk_size=args.mc_chunk_size,
            stride=args.mc_stride,
            top_k=args.mc_top_k,
            detach_memory=(args.dataset_code == 'xlong')
        )
        
        self.bias = nn.Parameter(torch.zeros(args.num_items + 1))

    def forward(self, x, embedding_weight, mask, labels=None):
        augmented = self.ssc(x, mask)
        
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

class SSCWrapper(nn.Module):
    def __init__(self, local_encoder, hidden_size, chunk_size, stride, top_k=2, detach_memory=True):
        super().__init__()
        self.local_encoder = local_encoder
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.stride = stride
        self.top_k = top_k
        self.detach_memory = detach_memory
        
        self.w_u = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_sequence, mask_sequence=None):
        B, L, D = x_sequence.size()
        outputs = []
        
        # 1. Tính toán Padding
        N = (L + self.stride - 1) // self.stride
        req_L = (N - 1) * self.stride + self.chunk_size
        pad_size = req_L - L
        
        if pad_size > 0:
            x_padded = F.pad(x_sequence, (0, 0, 0, pad_size))
            m_padded = F.pad(mask_sequence, (0, pad_size)) if mask_sequence is not None else None
        else:
            x_padded = x_sequence
            m_padded = mask_sequence
            
        # 2. Unfold để tạo Mega Batch cho tất cả các chunks
        # [B, D, N, chunk_size] -> [B, N, chunk_size, D]
        x_unfolded = x_padded.unfold(1, self.chunk_size, self.stride).permute(0, 2, 3, 1).contiguous()
        x_flat = x_unfolded.view(B * N, self.chunk_size, D)
        
        # 3. Chạy qua Local Encoder 1 LẦN DUY NHẤT cho tất cả chunks
        if m_padded is not None:
            m_unfolded = m_padded.unfold(1, self.chunk_size, self.stride).contiguous()
            m_flat = m_unfolded.view(B * N, self.chunk_size)
            h_flat = self.local_encoder(x_flat, m_flat)
        else:
            h_flat = self.local_encoder(x_flat)
            
        h_all_chunks = h_flat.view(B, N, self.chunk_size, D)
        
        # 4. Tiền tính toán Query và Online Scores song song
        u_all_chunks = self.w_u(h_all_chunks) # [B, N, chunk_size, D]
        
        cum_sum_all = torch.cumsum(h_all_chunks, dim=2)
        lengths_all = torch.arange(1, self.chunk_size+1, device=x_sequence.device, dtype=x_sequence.dtype).view(1, 1, self.chunk_size, 1)
        cum_mean_all = cum_sum_all / lengths_all
        online_scores_all = torch.sum(u_all_chunks * cum_mean_all, dim=-1, keepdim=True) / (self.hidden_size ** 0.5) # [B, N, chunk_size, 1]
        
        # 5. Khởi tạo Memory Buffers
        if N > 0:
            memory_tensor = torch.empty((B, N, D), device=x_sequence.device, dtype=x_sequence.dtype)
            mean_pool_tensor = torch.empty((B, N, D), device=x_sequence.device, dtype=x_sequence.dtype)
        num_mem = 0
        prev_end = 0
        
        # 6. Micro-loop: Chỉ chạy SSC Routing
        for chunk_idx, c_start in enumerate(range(0, L, self.stride)):
            c_end = min(c_start + self.chunk_size, L)
            S = c_end - c_start
            
            # Trích xuất dữ liệu của chunk hiện tại đã tính sẵn
            h_chunk = h_all_chunks[:, chunk_idx, :S, :]
            u_chunk = u_all_chunks[:, chunk_idx, :S, :]
            online_scores = online_scores_all[:, chunk_idx, :S, :]
            
            if num_mem > 0:
                past_mems = memory_tensor[:, :num_mem, :].clone()
                past_means = mean_pool_tensor[:, :num_mem, :].clone()
                
                past_scores = torch.bmm(u_chunk, past_means.transpose(1, 2)) / (self.hidden_size ** 0.5)
                
                topk_k = min(self.top_k, num_mem)
                if num_mem > topk_k:
                    topk_vals, topk_idx = torch.topk(past_scores, topk_k, dim=-1)
                    mask = torch.full_like(past_scores, float('-inf'))
                    past_scores = mask.scatter_(-1, topk_idx, topk_vals)
                
                joint_scores = torch.cat([online_scores, past_scores], dim=-1)
                joint_probs = F.softmax(joint_scores, dim=-1)
                
                online_prob = joint_probs[:, :, 0:1]
                past_probs = joint_probs[:, :, 1:]
                
                agg_past = torch.bmm(past_probs, past_mems)
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
                if self.detach_memory:
                    memory_tensor[:, num_mem, :] = h_tilde_chunk[:, -1, :].detach()
                    mean_pool_tensor[:, num_mem, :] = h_chunk.mean(dim=1).detach()
                else:
                    memory_tensor[:, num_mem, :] = h_tilde_chunk[:, -1, :]
                    mean_pool_tensor[:, num_mem, :] = h_chunk.mean(dim=1)
                num_mem += 1
                
        return torch.cat(outputs, dim=1)
