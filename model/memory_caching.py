import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryCaching(nn.Module):
    """
    Sparse Selective Caching (SSC) Memory Caching.
    
    Chia sequence thành overlapping chunks, cache hidden state cuối mỗi chunk,
    rồi dùng router chọn top-k cached memories phù hợp nhất cho mỗi token.
    
    Paper: Eq. 16-17 (Section 3.3)
    """
    def __init__(self, hidden_size, chunk_size=10, stride=5, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.stride = stride
        self.top_k = top_k
        
        # W_u: connector projection (Eq. 16)
        self.w_u = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, D) - output từ LRULayer
            mask: (B, L) - padding mask (not strictly needed for caching logic here, but passed for compatibility)
        Returns:
            (B, L, D) - output sau SSC aggregation
        """
        B, L, D = x.size()
        outputs = []
        
        # 1. Padding để x chia hết cho stride và chunk_size
        N = (L + self.stride - 1) // self.stride
        req_L = (N - 1) * self.stride + self.chunk_size
        pad_size = req_L - L
        
        if pad_size > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_size))
        else:
            x_padded = x
            
        # 2. Unfold để tạo Mega Batch cho tất cả các chunks
        # [B, L, D] -> [B, N, D, chunk_size] -> [B, N, chunk_size, D]
        x_unfolded = x_padded.unfold(1, self.chunk_size, self.stride).permute(0, 1, 3, 2).contiguous()
        h_all_chunks = x_unfolded # Ở LRU, x đã là hidden states nên ta dùng luôn làm h_all_chunks
        
        # 3. Tiền tính toán Query và Online Scores song song cho TẤT CẢ chunks
        u_all_chunks = self.w_u(h_all_chunks) # [B, N, chunk_size, D]
        
        cum_sum_all = torch.cumsum(h_all_chunks, dim=2)
        lengths_all = torch.arange(1, self.chunk_size+1, device=x.device, dtype=x.dtype).view(1, 1, self.chunk_size, 1)
        cum_mean_all = cum_sum_all / lengths_all
        online_scores_all = torch.sum(u_all_chunks * cum_mean_all, dim=-1, keepdim=True) / (self.hidden_size ** 0.5) # [B, N, chunk_size, 1]
        
        # 4. Khởi tạo Memory Buffers
        if N > 0:
            memory_tensor = torch.empty((B, N, D), device=x.device, dtype=x.dtype)
            mean_pool_tensor = torch.empty((B, N, D), device=x.device, dtype=x.dtype)
        num_mem = 0
        prev_end = 0
        
        # 5. Micro-loop: Chỉ chạy SSC Routing (rất nhẹ)
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
                
                # Tính past scores với tensor shapes: u_chunk(B, S, D) bmm past_means(B, N', D)^T -> (B, S, N')
                past_scores = torch.bmm(u_chunk, past_means.transpose(1, 2)) / (self.hidden_size ** 0.5)
                
                # Top-k selection
                topk_k = min(self.top_k, num_mem)
                if num_mem > topk_k:
                    topk_vals, topk_idx = torch.topk(past_scores, topk_k, dim=-1)
                    mask_score = torch.full_like(past_scores, float('-inf'))
                    past_scores = mask_score.scatter_(-1, topk_idx, topk_vals)
                
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
                memory_tensor[:, num_mem, :] = h_tilde_chunk[:, -1, :].detach()
                mean_pool_tensor[:, num_mem, :] = h_chunk.mean(dim=1).detach()
                num_mem += 1
                
        return torch.cat(outputs, dim=1)
