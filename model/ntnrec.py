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
                samples = torch.randint(1, self.args.num_items+1, size=(*augmented.shape[:2], num_samples,))
                all_items = torch.cat([samples.to(labels.device), labels.unsqueeze(-1)], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b l i d -> b l i', augmented, sampled_embeddings) + self.bias[all_items]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_
            else:
                num_samples = self.args.xlong_negative_sample_size
                samples = torch.randint(1, self.args.num_items+1, size=(augmented.shape[0], num_samples,))
                all_items = torch.cat([samples.to(labels.device), labels], dim=-1)
                sampled_embeddings = embedding_weight[all_items]
                scores = torch.einsum('b l d, b i d -> b l i', augmented, sampled_embeddings) + self.bias[all_items.unsqueeze(1)]
                labels_ = (torch.ones(labels.shape).long() * num_samples).to(labels.device)
                return scores, labels_.reshape(labels.shape)

class SSCModule(nn.Module):
    def __init__(self, hidden_size, chunk_size, top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.top_k = top_k
        
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

    def forward(self, h_sequence):
        B, L, D = h_sequence.size()
        outputs = []
        memory_buffer = []
        
        for t in range(L):
            h_t = h_sequence[:, t, :]
            
            if len(memory_buffer) > 0:
                M_tensor = torch.stack(memory_buffer, dim=1)
                
                q_t = self.w_q(h_t).unsqueeze(1)
                K = self.w_k(M_tensor)
                V = self.w_v(M_tensor)
                
                scores = torch.bmm(q_t, K.transpose(1, 2)) / (self.hidden_size ** 0.5)
                
                if M_tensor.size(1) > self.top_k:
                    topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
                    mask = torch.full_like(scores, float('-inf'))
                    scores = mask.scatter_(-1, topk_idx, topk_vals)
                    
                attn_weights = F.softmax(scores, dim=-1)
                agg_ssc = torch.bmm(attn_weights, V).squeeze(1)
                
                h_tilde = h_t + agg_ssc
            else:
                h_tilde = h_t
                
            outputs.append(h_tilde)
            
            if (t + 1) % self.chunk_size == 0:
                memory_buffer.append(h_tilde.detach())
                
        return torch.stack(outputs, dim=1)
