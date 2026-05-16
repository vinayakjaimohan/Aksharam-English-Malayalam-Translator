
import json
import torch
import unicodedata
from datasets import load_dataset
from tqdm import tqdm


# -------------------------------------------------
# Malayalam Unicode normalization
# -------------------------------------------------
def normalize_malayalam(text):
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFC", text)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    en_sentences = []
    ml_sentences = []

    print("\nCollecting datasets...")

    # -------------------------------------------------
    # 1. OPUS-100
    # -------------------------------------------------
    try:
        opus = load_dataset("opus100", "en-ml", split="train[:700]")

        before = len(en_sentences)

        for item in tqdm(opus, desc="OPUS-100"):

            t = item.get("translation", {})

            en = t.get("en")
            ml = t.get("ml")

            if isinstance(en, str) and isinstance(ml, str):

                ml = normalize_malayalam(ml)

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
            split="train[:100]"
        )

        before = len(en_sentences)

        for item in tqdm(hemanth, desc="Hemanth dataset"):

            en = item.get("english") or item.get("en")
            ml = item.get("malayalam") or item.get("ml")

            if isinstance(en, str) and isinstance(ml, str):

                ml = normalize_malayalam(ml)

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
            split="train[:50]"
        )

        before = len(en_sentences)

        for item in tqdm(samanantar, desc="Samanantar dataset"):

            en = item.get("src") or item.get("source")
            ml = item.get("tgt") or item.get("target")

            if isinstance(en, str) and isinstance(ml, str):

                ml = normalize_malayalam(ml)

                en_sentences.append(en.strip())
                ml_sentences.append(ml.strip())

        after = len(en_sentences)
        print(f"Samanantar added: {after - before}")

    except Exception as e:
        print("Samanantar dataset failed:", e)


    print("\nTotal collected pairs:", len(en_sentences))


    # -------------------------------------------------
    # Convert to dataset format
    # -------------------------------------------------

    dataset = []

    for i in range(len(en_sentences)):

        dataset.append({
            "en": en_sentences[i],
            "ml": ml_sentences[i]
        })


    # -------------------------------------------------
    # Save dataset
    # -------------------------------------------------

    with open("aligned_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("Dataset saved to aligned_dataset.json")


if __name__ == "__main__":
    main()

