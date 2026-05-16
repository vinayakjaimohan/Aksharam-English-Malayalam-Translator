import os
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
MODEL_PATH = "./aksharam_model_final"   # your trained model
OUTPUT_FILE = "translations.txt"


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Loading tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(BASE_MODEL)
    tokenizer.src_lang = "en_XX"

    print("Loading base model...")
    base_model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    model.to(device)
    model.eval()

    print("Model loaded successfully!\n")
    return model, tokenizer, device


# -----------------------------
# TRANSLATE FUNCTION
# -----------------------------
def translate(texts, model, tokenizer, device):
    tokenizer.src_lang = "en_XX"
    forced_bos_token_id = tokenizer.lang_code_to_id["ml_IN"]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    model, tokenizer, device = load_model()

    print("=" * 50)
    print("Aksharam Translator Ready!")
    print("Type 'exit' to quit")
    print("=" * 50)

    while True:
        try:
            text = input("\nEnter English sentence: ")

            if text.lower() in ["exit", "quit", "q"]:
                break

            if not text.strip():
                continue

            result = translate([text], model, tokenizer, device)[0]

            print("\nMalayalam:", result)

            # Save to file
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"{text} -> {result}\n")

        except KeyboardInterrupt:
            break

    print("\nExited.")


if __name__ == "__main__":
    main()