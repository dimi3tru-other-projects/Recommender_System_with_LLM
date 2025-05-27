"""
Batch-download and vectorize product images at 224x224 resolution.

Asynchronously downloads images from URLs in the dataset, processes each image by
resizing (maintaining aspect ratio) and padding to a 224x224 RGB square, and saves
them locally. Then, in batches, extracts feature embeddings using three vision
models—Google ViT Huge (patch14), Microsoft ResNet-50, and OpenAI CLIP ViT-L/14—
via Hugging Face Transformers on GPU/CPU/MPS. The script leverages aiohttp and
asyncio for concurrent downloads, ThreadPoolExecutor for CPU-bound image I/O and
preprocessing, and PyTorch for efficient model inference. Progress is tracked with
tqdm, and intermediate results are periodically saved to Parquet files.
"""

from typing import Optional, List, Dict
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import aiohttp
import asyncio
from io import BytesIO
from PIL import Image, ImageOps
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, ResNetModel, CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

import warnings
warnings.filterwarnings('ignore')

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ PATHS ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

file_path = Path().resolve() # Path(__file__).resolve()
project_dir = file_path
raw_data_path = project_dir / 'data' / 'raw'
products_data_dir = project_dir / 'data' / 'processed' / 'products_data'
processed_images_dir = products_data_dir / 'processed_images_224x224'
products_data_file = products_data_dir / 'products_data.parquet'
raw_data_path.mkdir(parents=True, exist_ok=True)
products_data_dir.mkdir(parents=True, exist_ok=True)
processed_images_dir.mkdir(parents=True, exist_ok=True)

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ MODELS ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

# Google ViT Huge
model_name_google_vit_huge_patch14_224_in21k = 'google/vit-huge-patch14-224-in21k'
processor_google_vit_huge_patch14_224_in21k = ViTImageProcessor.from_pretrained(model_name_google_vit_huge_patch14_224_in21k)
model_google_vit_huge_patch14_224_in21k = ViTModel.from_pretrained(model_name_google_vit_huge_patch14_224_in21k).eval().to(device)

# Microsoft ResNet-50
model_name_microsoft_resnet50 = 'microsoft/resnet-50'
processor_microsoft_resnet50 = AutoImageProcessor.from_pretrained(model_name_microsoft_resnet50)
model_microsoft_resnet50 = ResNetModel.from_pretrained(model_name_microsoft_resnet50).eval().to(device)

# OpenAI CLIP ViT-L/14
model_name_openai_clip_vit_large_patch14 = 'openai/clip-vit-large-patch14'
processor_openai_clip_vit_large_patch14 = CLIPProcessor.from_pretrained(model_name_openai_clip_vit_large_patch14)
model_openai_clip_vit_large_patch14 = CLIPModel.from_pretrained(model_name_openai_clip_vit_large_patch14).eval().to(device)
# Also there are other version of these models (another size or fitted versions, like timm)
# Patch16 (14x14 pixels) → 196 patches (since 224/14 = 16, and 16x16=256) => 256 tokens

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ FUNCTIONS ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

async def download_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """
    Asynchronously download an image from the specified URL.
    
    This function attempts to download an image using the provided aiohttp session.
    It handles timeouts and other exceptions, returning None in case of failure.
    
    Args:
        session: An active aiohttp ClientSession for making HTTP requests
        url: The URL of the image to download
        
    Returns:
        bytes: The binary content of the image if the download was successful
        None: If the download failed or the response status was not 200
        
    Raises:
        No exceptions are raised as they are caught and logged internally
    """
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.read()
    except Exception as e:
        print(f'Error downloading image from {url}: {e}')
    return None


def process_and_save_image(image_bytes: bytes, image_file: Path) -> None:
    """
    Process and save an image with standardized dimensions.
    
    This function takes raw image bytes, processes them to create a 224x224 RGB image,
    and saves it to the specified file path. The processing includes:
    1. Converting to RGB format
    2. Scaling the image proportionally to fit within 224x224
    3. Adding padding to create an exact 224x224 square image
    
    Args:
        image_bytes: Raw binary data of the image
        image_file: Path object representing the destination file location
        
    Returns:
        None
        
    Raises:
        No exceptions are raised as they are caught and logged internally
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        # Scaling it while maintaining proportions until it fits into 224x224
        image.thumbnail((224, 224), Image.Resampling.LANCZOS)
        # Adding padding to make it an exact 224x224 square
        image = ImageOps.pad(image, (224, 224), color=(0, 0, 0))
        image.save(image_file)
    except Exception as e:
        print(f'Error processing and saving image {image_file}: {e}')


def batch_vectorize_images(image_files: List[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate embeddings for a batch of images using multiple vision models.
    
    This function processes multiple images and extracts feature vectors (embeddings) from three
    different vision models:
    1. Google's ViT Huge Patch14 224 in21k
    2. Microsoft's ResNet-50
    3. OpenAI's CLIP ViT Large Patch14
    
    For each model, different embedding types are extracted (where applicable):
    - 'cls': The CLS token embedding
    - 'mean_patch': Average of all patch embeddings
    - 'pooled': The pooled representation (model-specific pooling)
    
    Args:
        image_files: List of Path objects pointing to image files to process
        
    Returns:
        A nested dictionary with structure:
        {
            'model_name': {
                'embedding_type': numpy_array_of_embeddings,
                ...
            },
            ...
        }
        Where each numpy array has shape (batch_size, embedding_dimension)
        
    Raises:
        No exceptions are raised as they are caught and logged internally
    """
    embeddings_dict = {}
    try:
        images = [Image.open(file).convert('RGB') for file in image_files]
        inputs_google_vit_huge_patch14_224_in21k = processor_google_vit_huge_patch14_224_in21k(
            images=images, do_resize=False, do_rescale=True, do_normalize=True, do_convert_rgb=False, rescale_factor=1/255, return_tensors='pt'
        ).to(device)
        inputs_microsoft_resnet50 = processor_microsoft_resnet50(
            images=images, do_resize=False, do_rescale=True, do_normalize=True, do_convert_rgb=False, rescale_factor=1/255, return_tensors='pt'
        ).to(device)
        inputs_openai_clip_vit_large_patch14 = processor_openai_clip_vit_large_patch14(
            images=images, do_resize=False, do_rescale=True, do_normalize=True, do_convert_rgb=False, rescale_factor=1/255, return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs_google_vit_huge_patch14_224_in21k = model_google_vit_huge_patch14_224_in21k(**inputs_google_vit_huge_patch14_224_in21k)
            outputs_microsoft_resnet50 = model_microsoft_resnet50(**inputs_microsoft_resnet50)
            outputs_openai_clip_vit_large_patch14 = model_openai_clip_vit_large_patch14.vision_model(**inputs_openai_clip_vit_large_patch14)

        LHS_google_vit_huge_patch14_224_in21k = outputs_google_vit_huge_patch14_224_in21k.last_hidden_state # (B, 257, 1280), 257 tokens: CLS + 256 patch tokens (224/14 = 16, 16×16 = 256)
        CLS_google_vit_huge_patch14_224_in21k = LHS_google_vit_huge_patch14_224_in21k[:, 0, :] # (B, 1280)
        mean_patch_google_vit_huge_patch14_224_in21k = LHS_google_vit_huge_patch14_224_in21k[:, 1:, :].mean(dim=1) # (B, 1280)
        pooled_google_vit_huge_patch14_224_in21k = outputs_google_vit_huge_patch14_224_in21k.pooler_output # (B, 1280), CLS token passed through a small feedforward layer (often linear + tanh)
        embeddings_dict['google/vit-huge-patch14-224-in21k'] = {
            'cls': CLS_google_vit_huge_patch14_224_in21k.cpu().numpy(), 
            'mean_patch': mean_patch_google_vit_huge_patch14_224_in21k.cpu().numpy(), 
            'pooled': pooled_google_vit_huge_patch14_224_in21k.cpu().numpy()
        }
        
        pooled_microsoft_resnet50 = outputs_microsoft_resnet50.pooler_output.squeeze(-1).squeeze(-1) # Global Avg Pooling of last hidden state (B, 2048, 7, 7) -> (B, 2048)
        embeddings_dict['microsoft/resnet-50'] = {'pooled': pooled_microsoft_resnet50.cpu().numpy()}

        LHS_openai_clip_vit_large_patch14 = outputs_openai_clip_vit_large_patch14.last_hidden_state # (B, 257, 1024)
        CLS_openai_clip_vit_large_patch14 = LHS_openai_clip_vit_large_patch14[:, 0, :] # (B, 1024)
        mean_patch_openai_clip_vit_large_patch14 = LHS_openai_clip_vit_large_patch14[:, 1:, :].mean(dim=1) # (B, 1024)
        with torch.no_grad():
            pooled_openai_clip_vit_large_patch14 = model_openai_clip_vit_large_patch14.visual_projection(CLS_openai_clip_vit_large_patch14) # (B, 768)
        
        embeddings_dict['openai/clip-vit-large-patch14'] = {
            'cls': CLS_openai_clip_vit_large_patch14.cpu().numpy(), 
            'mean_patch': mean_patch_openai_clip_vit_large_patch14.cpu().numpy(), 
            'pooled': pooled_openai_clip_vit_large_patch14.cpu().numpy()
        }

        return embeddings_dict
    except Exception as e:
        print(f'Error in batch vectorization: {e}')
        return embeddings_dict
    

async def process_batch(batch_indices, df, session, executor):
    loop = asyncio.get_running_loop()
    
    download_tasks = []
    for idx in batch_indices:
        url = df.at[idx, 'photo_analytics']
        image_file = processed_images_dir / f'image_{idx}_224x224.jpg'
        if not image_file.exists():
            download_tasks.append((idx, url, image_file))
    
    if download_tasks:
        coros = [download_image(session, url) for (_, url, _) in download_tasks]
        download_results = await asyncio.gather(*coros)
        save_tasks = []
        for (idx, url, image_file), image_bytes in zip(download_tasks, download_results):
            if image_bytes:
                save_tasks.append(loop.run_in_executor(executor, process_and_save_image, image_bytes, image_file))
            else:
                print(f'Failed to download image for index {idx}')
        if save_tasks:
            await asyncio.gather(*save_tasks)
    
    products_to_vectorize = []
    image_files = []
    for idx in batch_indices:
        if pd.isna(df.at[idx, 'image_path']):
            image_file = processed_images_dir / f'image_{idx}_224x224.jpg'
            if image_file.exists():
                products_to_vectorize.append(idx)
                image_files.append(image_file)
    
    if products_to_vectorize:
        embeddings_dict = await loop.run_in_executor(executor, batch_vectorize_images, image_files)
        # For each product in this batch, extract the corresponding row for each model.
        for i, idx in enumerate(products_to_vectorize):
            df.at[idx, 'CLS_google_vit_huge_patch14_224_in21k'] = embeddings_dict['google/vit-huge-patch14-224-in21k']['cls'][i].tolist()
            df.at[idx, 'mean_patch_google_vit_huge_patch14_224_in21k'] = embeddings_dict['google/vit-huge-patch14-224-in21k']['mean_patch'][i].tolist()
            df.at[idx, 'pooled_google_vit_huge_patch14_224_in21k'] = embeddings_dict['google/vit-huge-patch14-224-in21k']['pooled'][i].tolist()
            df.at[idx, 'pooled_microsoft_resnet50'] = embeddings_dict['microsoft/resnet-50']['pooled'][i].tolist()
            df.at[idx, 'CLS_openai_clip_vit_large_patch14'] = embeddings_dict['openai/clip-vit-large-patch14']['cls'][i].tolist()
            df.at[idx, 'mean_patch_openai_clip_vit_large_patch14'] = embeddings_dict['openai/clip-vit-large-patch14']['mean_patch'][i].tolist()
            df.at[idx, 'pooled_openai_clip_vit_large_patch14'] = embeddings_dict['openai/clip-vit-large-patch14']['pooled'][i].tolist()

    for idx in batch_indices:
        image_file = processed_images_dir / f'image_{idx}_224x224.jpg'
        df.at[idx, 'image_path'] = str(image_file) if image_file.exists() else None

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ MAIN SCRIPT ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

async def main(df, batch_size=64):
    mask = df['pooled_openai_clip_vit_large_patch14'].isna() | (df['pooled_openai_clip_vit_large_patch14'].astype(str).str.strip() == '')
    unprocessed_indices = df.loc[mask].index.tolist()
    total = len(unprocessed_indices)

    if total == 0:
        print('The process is over, everything is done.')
        return
    
    connector = aiohttp.TCPConnector(limit_per_host=64)
    async with aiohttp.ClientSession(connector=connector) as session:
        with ThreadPoolExecutor(max_workers=8) as executor:
            save_every_n_batches = 500
            num_batches = (total + batch_size - 1) // batch_size
            for batch_idx, start in enumerate(tqdm(range(0, total, batch_size), desc='Processing batches')):
                batch_indices = unprocessed_indices[start:start + batch_size]
                await process_batch(batch_indices, df, session, executor) # in-place by index
                # print(f'Processed {min(start + batch_size, total)}/{total} products.')
                # await asyncio.sleep(5)
                if (batch_idx + 1) % save_every_n_batches == 0 or (batch_idx + 1 == num_batches):
                    df.to_parquet(products_data_file)
                    # print(f'Progress saved, a total of {df.shape[0] - total} items were processed ({total} left).')

if __name__ == '__main__':
    new_cols = ['image_path', 'CLS_google_vit_huge_patch14_224_in21k', 'mean_patch_google_vit_huge_patch14_224_in21k', 'pooled_google_vit_huge_patch14_224_in21k',
        'pooled_microsoft_resnet50', 'CLS_openai_clip_vit_large_patch14', 'mean_patch_openai_clip_vit_large_patch14', 'pooled_openai_clip_vit_large_patch14']

    if products_data_file.is_file():
        df = pd.read_parquet(products_data_file)
        for col in new_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].astype(object)
        asyncio.run(main(df))

    else:
        needed_columns = ['articul_encrypred', 'color_base', 'brand', 'ktt1', 'ktt2', 
                          'ktt3', 'ktt4', 'title', 'product_id', 'product_created_at', 
                          'slug', 'photo_analytics', 'net_price']
        df_raw = pd.read_csv(raw_data_path / 'full_orders_v6.csv', sep=None, engine='python', usecols=needed_columns)
        df_sales = df_raw.groupby('product_id', as_index=False)['net_price'].sum().rename(columns={'net_price': 'sales_total'})
        needed_columns.remove('net_price')
        df = df_raw[needed_columns].dropna(subset=['photo_analytics']).drop_duplicates(subset=['product_id'], ignore_index=True)
        df = df.merge(df_sales, on='product_id', how='left')
        df['sales_total'].fillna(0, inplace=True)
        df.sort_values('sales_total', ascending=False, inplace=True) 

        for col in new_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].astype(object)
        df.to_parquet(products_data_file)

        del df_raw, df_sales

        asyncio.run(main(df))
