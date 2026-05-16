"""
FULL AUTOMATED PIPELINE

Malayalam scanned PDF → OCR → mal.txt
English digital PDF → extraction → eng.txt
LaBSE → paragraph alignment → aligned_paragraphs.csv

Uses your original extraction methods with automation
"""

import os
import pytesseract
from pdf2image import convert_from_path
import fitz
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch


# ==============================
# CONFIGURATION
# ==============================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\Acer\Downloads\Release-25.12.0-0\poppler\Library\bin"

MAL_PDF = "malayalam_novel.pdf"
ENG_PDF = "novel.pdf"

MAL_TXT = "mal.txt"
ENG_TXT = "eng.txt"

OUTPUT_CSV = "aligned_paragraphs.csv"

MODEL_NAME = "sentence-transformers/LaBSE"
WINDOW_SIZE = 5


pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ==============================
# MALAYALAM OCR EXTRACTION
# ==============================

def extract_malayalam():

    print("\n=== Malayalam OCR Extraction ===")

    images = convert_from_path(
        MAL_PDF,
        dpi=400,
        poppler_path=POPPLER_PATH
    )

    all_text = []

    for i, image in enumerate(images, 1):

        print(f"OCR page {i}/{len(images)}")

        text = pytesseract.image_to_string(
            image,
            lang="mal",
            config="--psm 3"
        )

        all_text.append(text)

    combined = "\n\n".join(all_text)

    with open(MAL_TXT, "w", encoding="utf-8") as f:
        f.write(combined)

    print("Malayalam saved to:", MAL_TXT)


# ==============================
# ENGLISH EXTRACTION
# ==============================

def extract_english():

    print("\n=== English PDF Extraction ===")

    doc = fitz.open(ENG_PDF)

    with open(ENG_TXT, "w", encoding="utf-8") as f:

        for page in doc:

            text = page.get_text("text")

            f.write(text + "\n")

    doc.close()

    print("English saved to:", ENG_TXT)


# ==============================
# PARAGRAPH SPLIT
# ==============================

def read_paragraphs(path):

    with open(path, "r", encoding="utf-8") as f:

        text = f.read()

    paragraphs = [

        p.strip()
        for p in text.split("\n\n")
        if len(p.strip()) > 10

    ]

    return paragraphs


# ==============================
# ALIGNMENT
# ==============================

def align_paragraphs():

    print("\n=== LaBSE Alignment ===")

    eng_paragraphs = read_paragraphs(ENG_TXT)
    mal_paragraphs = read_paragraphs(MAL_TXT)

    print("English paragraphs:", len(eng_paragraphs))
    print("Malayalam paragraphs:", len(mal_paragraphs))


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Generating embeddings...")

    eng_emb = model.encode(
        eng_paragraphs,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    mal_emb = model.encode(
        mal_paragraphs,
        convert_to_tensor=True,
        show_progress_bar=True
    )


    sim = cos_sim(eng_emb, mal_emb).cpu().numpy()


    print("Running sequential alignment...")

    aligned = []

    mal_pointer = 0

    for i in range(len(eng_paragraphs)):

        start = mal_pointer
        end = min(len(mal_paragraphs), mal_pointer + WINDOW_SIZE)

        best_score = -1
        best_j = None

        for j in range(start, end):

            score = sim[i][j]

            if score > best_score:

                best_score = score
                best_j = j

        if best_j is not None:

            aligned.append({

                "English": eng_paragraphs[i],
                "Malayalam": mal_paragraphs[best_j],
                "Similarity": float(best_score)

            })

            mal_pointer = best_j


    df = pd.DataFrame(aligned)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Alignment saved to:", OUTPUT_CSV)


# ==============================
# MAIN
# ==============================

def main():

    extract_malayalam()

    extract_english()

    align_paragraphs()


if __name__ == "__main__":

    main()
