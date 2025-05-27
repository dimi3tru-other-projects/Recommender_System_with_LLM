import os
from pathlib import Path
import asyncio
import pandas as pd
import httpx
from tqdm.asyncio import tqdm
from typing import List, Tuple, Optional
from dotenv import load_dotenv
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ CONFIG ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(f"OPENROUTER_API_KEY not found in {ENV_PATH}")
MODEL = "google/gemini-2.0-flash-lite-001"
PATH = Path("data/processed/products_data/products_data.parquet")
BATCH_SIZE = 512
PARALLEL_LIMIT = 128 # Number of parallel requests
OVERWRITE = False # If True, regenerate descriptions for all items

SYSTEM_PROMPT = (
    "You generate clear, detailed product descriptions for e-commerce based on product images. "
    "Write in English, using complete sentences. Focus only on visible attributes and key features. "
    "Do not use subjective words or marketing language. Keep the style factual and informative, as in an Amazon product listing."
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ FUNCTIONS ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

def build_user_prompt(row: pd.Series) -> str:
    """
    Build a user prompt for the model from a pandas Series row.

    Args:
        row (pd.Series): A single row of product data.

    Returns:
        str: A structured text prompt describing the product.
    """
    categories = " > ".join(
        str(row[c]) for c in ("ktt1", "ktt2", "ktt3", "ktt4") if pd.notna(row[c])
    )
    return (
        f"Brand: {row['brand']}\n"
        f"Color: {row['color_base']}\n"
        f"Categories: {categories}\n"
        f"Title: {row['title']}\n"
        f"Created on: {row['product_created_at']}"
    )


async def generate_description_openrouter_async(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_prompt: str,
    image_url: str,
    max_retries: int = 3,
    backoff: float = 2.0
) -> Optional[str]:
    """
    Make an async request to OpenRouter API to generate a product description.

    Args:
        client (httpx.AsyncClient): The HTTP client for requests.
        system_prompt (str): System prompt text.
        user_prompt (str): User prompt text.
        image_url (str): URL of the product image.
        max_retries (int): Maximum retries for the request.
        backoff (float): Exponential backoff base.

    Returns:
        Optional[str]: Generated description or None if all attempts fail.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.post(API_URL, headers=HEADERS, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                raise KeyError("missing choices")
            first = choices[0].get("message", {})
            content = first.get("content")
            if content and isinstance(content, str) and content.strip():
                return content.strip()
            raise KeyError("empty content")
        except Exception as e:
            if attempt == max_retries:
                return None
            await asyncio.sleep(backoff ** (attempt - 1))


async def process_rows(
    df: pd.DataFrame,
    indices: List[int],
    sema: asyncio.Semaphore
) -> List[Tuple[int, Optional[str]]]:
    """
    Process a batch of rows asynchronously, generating product descriptions.

    Args:
        df (pd.DataFrame): The DataFrame with product data.
        indices (List[int]): Row indices to process.
        sema (asyncio.Semaphore): Semaphore to limit concurrency.

    Returns:
        List[Tuple[int, Optional[str]]]: List of (index, description) results.
    """
    async with httpx.AsyncClient() as client:
        async def process_one(idx: int) -> Tuple[int, Optional[str]]:
            async with sema:
                row = df.loc[idx]
                prompt = build_user_prompt(row)
                img_url = row["photo_analytics"]
                desc = await generate_description_openrouter_async(
                    client, SYSTEM_PROMPT, prompt, img_url
                )
                return idx, desc
        coros = [process_one(idx) for idx in indices]
        results = await tqdm.gather(*coros)
    return results


def get_pending_indices(df: pd.DataFrame) -> List[int]:
    """
    Get a list of row indices that need description generation.

    Args:
        df (pd.DataFrame): DataFrame with product data.

    Returns:
        List[int]: Indices for processing.
    """
    if OVERWRITE or "description" not in df.columns:
        df["description"] = ""
    return [i for i, txt in enumerate(df["description"].astype(str)) if OVERWRITE or not txt.strip()]

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ MAIN SCRIPT ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

async def main() -> None:
    """
    Main asynchronous batch-processing routine for generating product descriptions.
    Supports overwrite and automatic resume.
    """
    # Load data
    df = pd.read_parquet(PATH)
    df = df.dropna(subset=["photo_analytics"]).reset_index(drop=True)
    pending = get_pending_indices(df)
    print(f"{len(pending)} items pending")

    if not pending:
        print("Nothing to do.")
        return

    sema = asyncio.Semaphore(PARALLEL_LIMIT)
    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_iter, start in enumerate(tqdm(range(0, len(pending), BATCH_SIZE), desc="Processing batches"), 1):
        batch = pending[start:start+BATCH_SIZE]
        results = await process_rows(df, batch, sema)
        for idx, desc in results:
            if desc:
                df.at[idx, "description"] = desc
        # Save every 64 batches or on the last batch
        if (batch_iter % 64 == 0) or (batch_iter == total_batches):
            df.to_parquet(PATH)
    print("All done. Final file:", PATH)

if __name__ == "__main__":
    asyncio.run(main())
