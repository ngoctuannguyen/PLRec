import csv
import json
import argparse
import os

def build(dataset_code):
    path = f'data/{dataset_code}/item_description.csv'
    if not os.path.exists(path):
        print(f"Error: Could not find {path}")
        return
        
    meta = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            item_id = row['item_id:token']
            meta[item_id] = {
                'title': row.get('title:token', ''),
                'category': row.get('categories:token_seq', ''),
                'brand': row.get('brand:token', ''),
                'price': row.get('price:float', ''),
                'sales_type': row.get('sales_type:token', ''),
            }
            
    out_path = f'data/{dataset_code}/item_meta.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        
    print(f'Saved {len(meta)} items to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='beauty')
    args = parser.parse_args()
    build(args.dataset)
