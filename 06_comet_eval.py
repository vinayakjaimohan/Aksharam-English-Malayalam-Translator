import os
import json
import random
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
from comet import download_model, load_from_checkpoint

# -----------------------------
# FIX WINDOWS SYMLINK ISSUE
# -----------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
MODEL_PATH = "./aksharam_model_final"
DATA_PATH = "final_dataset.json"

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# LOAD TRANSLATION MODEL
# -----------------------------
print("\nLoading translation model...")

tokenizer = MBart50TokenizerFast.from_pretrained(BASE_MODEL)
tokenizer.src_lang = "en_XX"

base_model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

model.to(device)
model.eval()

print("Translation model ready!")

# -----------------------------
# LOAD DATASET (AUTO)
# -----------------------------
print("\nLoading dataset...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Take random 50 samples
sample = random.sample(dataset, 50)

data = []
for item in sample:
    data.append({
        "src": item["en"],
        "ref": item["ml"]
    })

print(f"Using {len(data)} samples for evaluation")

# -----------------------------
# TRANSLATE FUNCTION
# -----------------------------
def translate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
    ).to(device)

    forced_bos_token_id = tokenizer.lang_code_to_id["ml_IN"]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=5,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# GENERATE TRANSLATIONS
# -----------------------------
print("\nGenerating translations...\n")

for item in data:
    item["mt"] = translate(item["src"])
    print(f"{item['src']} → {item['mt']}")

# -----------------------------
# LOAD COMET MODEL (LATEST)
# -----------------------------
print("\nDownloading COMET model (first time only)...")

model_path = download_model("Unbabel/wmt22-comet-da")

print("Loading COMET model...")
comet_model = load_from_checkpoint(model_path)

print("COMET model ready!")

# -----------------------------
# RUN COMET
# -----------------------------
print("\nRunning COMET evaluation...")

output = comet_model.predict(data, batch_size=8)

sys_score = output["system_score"]

print("\n🔥 FINAL COMET SCORE:", round(sys_score, 4))