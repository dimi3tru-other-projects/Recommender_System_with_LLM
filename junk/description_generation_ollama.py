"""
Batch-generate product descriptions using the Ollama Python client.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from ollama import chat

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

MODEL = "gemma3:27b-it-q4_K_M"
INPUT_PATH = Path("data/processed/products_data/products_data.parquet")
OUTPUT_PATH = Path("data/processed/products_data/products_data_with_desc.parquet")
OUTPUT_PATH = Path("data/processed/products_data/products_data_test.parquet")
BATCH_SIZE = 4
RESUME = True  # if True, skip rows that already have a non-empty description

# SYSTEM_PROMPT = (
#     "You are an assistant that writes short, objective product descriptions for e-commerce. "
#     "Given structured product metadata, generate a concise description suitable for vector-based semantic search. "
#     "Focus strictly on factual information. Do not include any subjective words "
#     "such as 'beautiful', 'amazing', or 'perfect'. Do not ask questions. "
#     "Do not mention internal fields or metadata labels."
# )

SYSTEM_PROMPT = (
    "You generate clear, detailed product descriptions for e-commerce based on product images. "
    "Write in English, using complete sentences. Focus only on visible attributes and key features. "
    "Do not use subjective words or marketing language. Keep the style factual and informative, as in an Amazon product listing."
)


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def generate_description_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_path: Optional[Path] = None,
    timeout: int = 600,
) -> Optional[str]:
    """
    Call Ollama via its Python client to generate one product description.

    Args:
        model: the Ollama model name
        system_prompt: global instruction
        user_prompt: per-item metadata prompt
        image_path: optional Path to an image file
        timeout: maximum seconds to wait (ignored by Ollama client)

    Returns:
        The generated text, or None on failure.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if image_path and image_path.is_file():
        # attach the image to the user message
        messages[-1]["images"] = [image_path]

    try:
        response = chat(model=model, messages=messages)
        # Ollama client returns {'message': {'role':'assistant','content':...}, ...}
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] Ollama chat failed for prompt '{user_prompt[:30]}...': {e}")
        return None


def build_user_prompt(row: pd.Series) -> str:
    """
    Construct the user prompt from product metadata for semantic description generation.

    Args:
        row: a pandas Series with keys 'brand', 'color_base', 'ktt1'...'ktt4',
             'title', 'product_created_at', and optional 'image_path'.

    Returns:
        A structured factual prompt.
    """
    categories = " > ".join(
        str(row[col]) for col in ("ktt1", "ktt2", "ktt3", "ktt4") if pd.notna(row[col])
    )
    return (
        f"Brand: {row['brand']}\n"
        f"Color: {row['color_base']}\n"
        f"Categories: {categories}\n"
        f"Title: {row['title']}\n"
        f"Created on: {row['product_created_at']}\n\n"
        "Generate a short factual description based on this metadata."
    )


def process_batch(df: pd.DataFrame, indices: list[int]) -> None:
    """
    Generate descriptions in-place for a batch of row indices.

    Args:
        df: DataFrame with a 'description' column (will be modified)
        indices: list of integer row indices to process
    """
    for idx in indices:
        existing = df.at[idx, "description"]
        if RESUME and isinstance(existing, str) and existing.strip():
            print(f"[SKIP] idx={idx} already has a description.")
            continue

        row = df.loc[idx]
        prompt = build_user_prompt(row)
        img_path = Path(row["image_path"]) if pd.notna(row.get("image_path")) else None

        desc = generate_description_ollama(MODEL, SYSTEM_PROMPT, prompt, img_path)
        df.at[idx, "description"] = desc or ""
        print(f"[DONE] idx={idx}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN SCRIPT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load existing output if resuming, else fresh input
    if RESUME and OUTPUT_PATH.exists():
        print(f"Resuming from {OUTPUT_PATH}")
        df = pd.read_parquet(OUTPUT_PATH)
    else:
        print(f"Loading fresh data from {INPUT_PATH}")
        df = pd.read_parquet(INPUT_PATH)
        if "description" not in df.columns:
            df["description"] = ""

    # 2) Drop rows without images and reset index
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # 3) Identify which indices still need descriptions
    mask_done = df["description"].astype(str).str.strip() != ""
    unprocessed = [i for i, done in enumerate(mask_done) if not done]

    if not unprocessed:
        print("All items already processed. Nothing to do.")
        return

    print(f"{len(unprocessed):,} items to process, in batches of {BATCH_SIZE}.")

    # 4) Batch over only unprocessed indices
    for start in tqdm(range(0, len(unprocessed), BATCH_SIZE), desc="Batches"):
        batch_idxs = unprocessed[start:start + BATCH_SIZE]
        process_batch(df, batch_idxs)
        print(f"Saving progress to {OUTPUT_PATH}")
        df.to_parquet(OUTPUT_PATH)

    print("Processing complete. Final data saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
