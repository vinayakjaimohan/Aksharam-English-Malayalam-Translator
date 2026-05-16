import json
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import time


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading LaBSE model...")
    # Load on CPU first to avoid initialization hang
    model = SentenceTransformer("sentence-transformers/LaBSE")
    model = model.to(device)
    model.eval()
    
    # Disable gradients to save memory
    torch.set_grad_enabled(False)
    
    print(f"Model loaded on: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

    en_sentences = []
    ml_sentences = []

    print("Collecting datasets...")

    # -------------------------------------------------
    # 1. OPUS-100
    # -------------------------------------------------
    try:
        opus = load_dataset("opus100", "en-ml", split="train[:40000]")

        before = len(en_sentences)

        for item in tqdm(opus, desc="OPUS-100"):

            t = item.get("translation", {})

            en = t.get("en")
            ml = t.get("ml")

            if isinstance(en, str) and isinstance(ml, str):
                en_sentences.append(en.strip())
                ml_sentences.append(ml.strip())

        after = len(en_sentences)
        print(f"OPUS added: {after - before}")

    except Exception as e:
        print("OPUS load failed:", e)


    # -------------------------------------------------
    # 2. Hemanth dataset
    # -------------------------------------------------
    try:
        hemanth = load_dataset(
            "Hemanth-thunder/english-to-malayalam-mt",
            split="train[:50000]"
        )

        before = len(en_sentences)

        for item in tqdm(hemanth, desc="Hemanth dataset"):

            en = item.get("english") or item.get("en")
            ml = item.get("malayalam") or item.get("ml")

            if isinstance(en, str) and isinstance(ml, str):
                en_sentences.append(en.strip())
                ml_sentences.append(ml.strip())

        after = len(en_sentences)
        print(f"Hemanth added: {after - before}")

    except Exception as e:
        print("Hemanth dataset failed:", e)


    # -------------------------------------------------
    # 3. Samanantar dataset
    # -------------------------------------------------
    try:
        samanantar = load_dataset(
            "ai4bharat/samanantar",
            "ml",
            split="train[:100000]"
        )

        before = len(en_sentences)

        for item in tqdm(samanantar, desc="Samanantar dataset"):

            en = item.get("src")
            ml = item.get("tgt")

            if isinstance(en, str) and isinstance(ml, str):
                en_sentences.append(en.strip())
                ml_sentences.append(ml.strip())

        after = len(en_sentences)
        print(f"Samanantar added: {after - before}")

    except Exception as e:
        print("Samanantar dataset failed:", e)


    print("\nTotal collected pairs:", len(en_sentences))


    # -------------------------------------------------
    # LaBSE semantic filtering (FAST CPU version)
    # -------------------------------------------------

    threshold = 0.60
    batch_size = 64  # Reduced for encoding efficiency
    aligned_pairs = []

    print("\nFiltering pairs using semantic similarity (vectorized on GPU)...")
    filter_start = time.time()
    
    for i in tqdm(range(0, len(en_sentences), batch_size)):

        batch_en = en_sentences[i:i + batch_size]
        batch_ml = ml_sentences[i:i + batch_size]

        # Encode with small batches (32 at a time for GPU stability)
        en_emb = model.encode(
            batch_en,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False,
            device=device
        )

        ml_emb = model.encode(
            batch_ml,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False,
            device=device
        )
        
        # Ensure tensors are on GPU for fast vectorized similarity
        en_emb = en_emb.to(device)
        ml_emb = ml_emb.to(device)
        
        # Fast vectorized cosine similarity (runs on GPU)
        scores = util.cos_sim(en_emb, ml_emb).diag()

        # Vectorized filtering (all at once, not loop)
        valid_mask = scores >= threshold
        
        for j in torch.where(valid_mask)[0]:
            aligned_pairs.append({
                "en": batch_en[int(j)],
                "ml": batch_ml[int(j)],
                "score": float(scores[j])
            })

    filter_time = time.time() - filter_start

    print(
        f"Kept {len(aligned_pairs)} high-quality pairs "
        f"out of {len(en_sentences)} ({filter_time:.1f}s)"
    )


    # -------------------------------------------------
    # Save dataset
    # -------------------------------------------------

    with open("aligned_dataset.json", "w", encoding="utf-8") as f:
        json.dump(aligned_pairs, f, ensure_ascii=False, indent=2)

    print("Dataset saved to aligned_dataset.json")


if __name__ == "__main__":
    main()