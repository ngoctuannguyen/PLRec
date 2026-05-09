import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm

from datasets import DATASETS
from config import args, set_template
from dataloader import dataloader_factory
from model.lru import LRU

def load_item_meta(dataset_code):
    path = f'data/{dataset_code}/item_meta.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def format_prompt(history_ids, candidate_ids, item_meta):
    """Build simplified prompt. Only title + category to reduce noise."""
    history_ids = history_ids[-10:]
    labels = [chr(ord('A') + i) for i in range(len(candidate_ids))]
    
    prompt = (
        "You are an expert recommender system. Analyze the user's historical purchases in CHRONOLOGICAL ORDER.\n"
    )
    
    prompt += "## History:\n"
    for i, item_id in enumerate(history_ids):
        meta = item_meta.get(str(item_id), {})
        title = meta.get('title', 'Unknown')
        category = meta.get('category', 'Unknown')
        prompt += f"{i+1}. {title} [{category}]\n"

    prompt += "\n## Candidates:\n"
    for label, item_id in zip(labels, candidate_ids):
        meta = item_meta.get(str(item_id), {})
        title = meta.get('title', 'Unknown')
        category = meta.get('category', 'Unknown')
        prompt += f"[{label}] {title} [{category}]\n"

    prompt += (
        "\nWhich candidate item will the user purchase next based on their history?\n"
        "Output ONLY the single letter of the item inside brackets (e.g., A, B).\n"
        "Answer: "
    )
    return prompt, labels

def main():
    args.dataset_code = 'games'
    set_template(args)
    
    # 1. Load Dataset — use TRAIN loader (sliding window = many subsequences)
    train_loader, _, _ = dataloader_factory(args)
    
    # 2. Load LRU Model (Retriever)
    model = LRU(args)
    export_root = f"experiments/{args.model_code}/{args.dataset_code}_{args.weight_decay}_{args.bert_dropout}_{args.bert_attn_dropout}"
    model_path = f"{export_root}/models/best_acc_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Best model not found at {model_path}. Please train LRU first.")
        return
        
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    item_meta = load_item_meta(args.dataset_code)
    
    # 3. Generate Data from TRAINING set (like LlamaRec)
    output_file = f"data/{args.dataset_code}/lora_train.jsonl"
    print(f"Generating data to {output_file}...")
    print(f"Train loader has {len(train_loader)} batches")
    
    num_samples = 0
    K = 20  # Top-K candidates (LlamaRec uses 20)
    
    with open(output_file, 'w', encoding='utf-8') as f, torch.no_grad():
        for batch in tqdm(train_loader, desc="Processing train batches"):
            seqs, labels_batch = batch
            seqs = seqs.to(device)
            labels_batch = labels_batch.to(device)
            
            # Predict scores for all items
            scores = model(seqs)
            if isinstance(scores, tuple):
                scores = scores[0]
            scores = scores[:, -1, :]  # Score for next item prediction
            
            B = seqs.size(0)
            for i in range(B):
                seq_items = seqs[i][seqs[i] > 0]
                scores[i, seq_items] = -1e9
                scores[i, 0] = -1e9
                
            # Get Top-K Candidates
            _, top_k_ids = torch.topk(scores, K, dim=-1)
            
            for i in range(B):
                history = seqs[i][seqs[i] > 0].cpu().tolist()
                candidates = top_k_ids[i].cpu().tolist()
                
                # The target is the LAST label (next item in sequence)
                # In SASTrainDataset: tokens = seq[:-1], labels = seq[1:]
                # So labels[-1] is the ground truth next item
                target_labels = labels_batch[i][labels_batch[i] > 0].cpu().tolist()
                if not target_labels or not history:
                    continue
                target = target_labels[-1]  # Last non-zero label = next item
                
                if target == 0:
                    continue
                    
                # Inject Ground Truth into candidates if missing
                if target not in candidates:
                    replace_idx = random.randint(0, K-1)
                    candidates[replace_idx] = target
                
                # Take only K candidates (in case of duplicates)
                candidates = list(dict.fromkeys(candidates))[:K]
                    
                # Shuffle candidates
                random.shuffle(candidates)
                
                # Format Prompt
                prompt, lbls = format_prompt(history, candidates, item_meta)
                
                # Find which label is the target
                if target not in candidates:
                    continue
                target_idx = candidates.index(target)
                target_label = lbls[target_idx]
                
                # Save as JSONL
                sample = {
                    "instruction": prompt,
                    "output": target_label + "]"
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                num_samples += 1
                
    print(f"Done! Generated {num_samples} instruction-tuning samples.")
    print(f"(Previously was ~24K from val set, now using full training set)")

if __name__ == '__main__':
    main()
