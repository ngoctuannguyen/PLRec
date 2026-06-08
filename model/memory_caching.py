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
        
        # Tiền tính toán connector cho toàn bộ sequence
        # u = x @ W_u: (B, L, D)
        u_all = self.w_u(x)
        
        num_mem = 0
        prev_end = 0
        
        # Tính N (số chunks tối đa)
        # Sẽ cấp phát memory buffer tối đa N chunks
        N = (L + self.stride - 1) // self.stride
        if N > 0:
            memory_tensor = torch.empty((B, N, D), device=x.device, dtype=x.dtype)
            mean_pool_tensor = torch.empty((B, N, D), device=x.device, dtype=x.dtype)
            
        for chunk_idx, c_start in enumerate(range(0, L, self.stride)):
            c_end = min(c_start + self.chunk_size, L)
            S = c_end - c_start
            
            # Trích xuất chunk hiện tại từ sequence đã tính sẵn
            h_chunk = x[:, c_start:c_end, :] # (B, S, D)
            u_chunk = u_all[:, c_start:c_end, :] # (B, S, D)
            
            # Tính cumulative mean cho online score
            # (B, S, D)
            cum_sum = torch.cumsum(h_chunk, dim=1)
            lengths = torch.arange(1, S + 1, device=x.device, dtype=x.dtype).view(1, S, 1)
            cum_mean = cum_sum / lengths
            
            # Tính online_score (self-reference)
            # ⟨u, cumulative_mean⟩ / sqrt(D)
            online_scores = torch.sum(u_chunk * cum_mean, dim=-1, keepdim=True) / (self.hidden_size ** 0.5) # (B, S, 1)
            
            if num_mem > 0:
                past_mems = memory_tensor[:, :num_mem, :].clone()
                past_means = mean_pool_tensor[:, :num_mem, :].clone()
                
                # Tính relevance scores cho các past chunks
                # r_i = ⟨u, mean_pool(chunk_i)⟩
                # u_chunk: (B, S, D), past_means: (B, N', D)
                # past_scores: (B, S, N')
                past_scores = torch.bmm(u_chunk, past_means.transpose(1, 2)) / (self.hidden_size ** 0.5)
                
                # Top-k selection
                topk_k = min(self.top_k, num_mem)
                if num_mem > topk_k:
                    topk_vals, topk_idx = torch.topk(past_scores, topk_k, dim=-1)
                    mask_score = torch.full_like(past_scores, float('-inf'))
                    past_scores = mask_score.scatter_(-1, topk_idx, topk_vals)
                
                # Joint softmax
                joint_scores = torch.cat([online_scores, past_scores], dim=-1) # (B, S, 1 + N')
                joint_probs = F.softmax(joint_scores, dim=-1)
                
                online_prob = joint_probs[:, :, 0:1] # (B, S, 1)
                past_probs = joint_probs[:, :, 1:] # (B, S, N')
                
                # Aggregate
                agg_past = torch.bmm(past_probs, past_mems) # (B, S, N') @ (B, N', D) -> (B, S, D)
                h_tilde_chunk = online_prob * h_chunk + agg_past
            else:
                h_tilde_chunk = h_chunk
                
            # Trích xuất các token mới để tránh trùng lặp
            if prev_end < c_end:
                start_idx_in_chunk = prev_end - c_start
                new_tokens = h_tilde_chunk[:, start_idx_in_chunk:, :]
                outputs.append(new_tokens)
                prev_end = c_end
                
            # Cập nhật Memory Buffers nếu chunk này hoàn chỉnh
            if S == self.chunk_size:
                # Lấy hidden state cuối cùng của aggregated chunk để cache (checkpoint)
                memory_tensor[:, num_mem, :] = h_tilde_chunk[:, -1, :].detach()
                # Tính mean pool từ original h_chunk
                mean_pool_tensor[:, num_mem, :] = h_chunk.mean(dim=1).detach()
                num_mem += 1
                
        return torch.cat(outputs, dim=1)
