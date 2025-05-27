"""
Batch-vectorize product descriptions using top English text embedding models.

Loads product descriptions from a Parquet file, computes vector embeddings in batches using:
  1. intfloat/e5-large-v2
  2. BAAI/bge-large-en-v1.5
  3. nomic-ai/nomic-embed-text-v1

Results are saved to new columns in the same Parquet file. The script uses PyTorch (CPU/GPU/MPS)
and supports interrupted/resume workflow. Periodic saving and tqdm progress bars included.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CONFIG ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

BATCH_SIZE = 32
products_data_file = Path().resolve() / 'data' / 'processed' / 'products_data' / 'products_data.parquet'
cols = [
    'embedding_e5_large_v2',
    'embedding_bge_large_en_v15',
    'embedding_nomic_embed_text_v15'
]

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ MODELS ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

tokenizer_e5 = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
model_e5 = AutoModel.from_pretrained('intfloat/e5-large-v2').eval().to(device)

tokenizer_bge = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
model_bge = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').eval().to(device)

tokenizer_nomic = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
model_nomic = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True).eval().to(device)

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ FUNCTIONS ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

def add_prefix_e5(texts: List[str]) -> List[str]:
    return [f"query: {t}" for t in texts]

def add_prefix_nomic(texts: List[str]) -> List[str]:
    return [f"search_document: {t}" for t in texts]

def batch_embed_e5(texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        texts = add_prefix_e5(texts)
        batch = tokenizer_e5(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        output = model_e5(**batch)
        last_hidden = output.last_hidden_state
        attn_mask = batch['attention_mask']
        last_hidden = last_hidden.masked_fill(~attn_mask[..., None].bool(), 0.0)
        emb = last_hidden.sum(dim=1) / attn_mask.sum(dim=1)[..., None]
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

def batch_embed_bge(texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        batch = tokenizer_bge(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        output = model_bge(**batch)
        emb = output[0][:, 0]  # CLS pooling
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

def batch_embed_nomic(texts: List[str], matryoshka_dim: int = 512) -> np.ndarray:
    with torch.no_grad():
        texts = add_prefix_nomic(texts)
        batch = tokenizer_nomic(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        output = model_nomic(**batch)
        attn_mask = batch['attention_mask']
        token_embeddings = output[0]
        input_mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
        pooled = pooled[:, :matryoshka_dim]
        emb = F.normalize(pooled, p=2, dim=1)
    return emb.cpu().numpy()

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ MAIN SCRIPT ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

def main():
    if not products_data_file.is_file():
        print(f"No products data file found at {products_data_file}.")
        return

    df = pd.read_parquet(products_data_file)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(object)

    mask = df['embedding_e5_large_v2'].isna() | (df['embedding_e5_large_v2'].astype(str).str.strip() == '')
    unprocessed_indices = df.loc[mask].index.tolist()
    total = len(unprocessed_indices)
    print(f"Total texts to process: {total}")
    if total == 0:
        print("All texts already embedded.")
        return

    save_every = 3000 
    for batch_num, batch_start in enumerate(tqdm(range(0, total, BATCH_SIZE), desc='Processing batches'), 1):
        batch_indices = unprocessed_indices[batch_start:batch_start+BATCH_SIZE]
        batch_texts = df.loc[batch_indices, 'description'].astype(str).tolist()

        e5_embeds = batch_embed_e5(batch_texts)
        bge_embeds = batch_embed_bge(batch_texts)
        nomic_embeds = batch_embed_nomic(batch_texts, matryoshka_dim=512)

        for i, idx in enumerate(batch_indices):
            df.at[idx, 'embedding_e5_large_v2'] = e5_embeds[i].tolist()
            df.at[idx, 'embedding_bge_large_en_v15'] = bge_embeds[i].tolist()
            df.at[idx, 'embedding_nomic_embed_text_v15'] = nomic_embeds[i].tolist()

        # Save every 3000 batches or at last batch
        if batch_num % save_every == 0 or (batch_start + BATCH_SIZE >= total):
            df.to_parquet(products_data_file)
            print(f'Saved at batch {batch_num}')

    print("Text embeddings completed and saved.")

if __name__ == '__main__':
    main()
