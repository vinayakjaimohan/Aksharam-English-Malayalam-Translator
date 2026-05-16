import os
import json
import torch
from datasets import Dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# -----------------------------
# FORCE GPU
# -----------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE")

torch.cuda.set_device(0)


# -----------------------------
# Load dataset
# -----------------------------
def load_aligned_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples.")
    return Dataset.from_list(data)


# -----------------------------
# Preprocess (FIXED)
# -----------------------------
def preprocess_function(examples, tokenizer):
    inputs = ["translate English to Malayalam: " + x for x in examples["en"]]

    model_inputs = tokenizer(
        inputs,
        max_length=64,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["ml"],
        max_length=64,
        truncation=True,
    )

    # 🔥 mask padding tokens
    labels_ids = labels["input_ids"]
    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in labels_ids
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs


# -----------------------------
# MAIN
# -----------------------------
def main():

    model_name = "google/mt5-small"
    data_path = "aligned_dataset.json"

    print("🔥 Using GPU:", torch.cuda.get_device_name(0))

    if not os.path.exists(data_path):
        print("Dataset not found")
        return

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = MT5Tokenizer.from_pretrained(model_name)

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = load_aligned_data(data_path)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print("Tokenizing dataset...")

    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        num_proc=2,
        remove_columns=dataset["train"].column_names
    )

    # -----------------------------
    # Model
    # -----------------------------
    print("Loading model...")
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # ❌ NO gradient checkpointing (simpler + stable)
    model.cuda()

    print("Model device:", next(model.parameters()).device)

    # -----------------------------
    # Data collator
    # -----------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest"
    )

    # -----------------------------
    # Training args (STABLE)
    # -----------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir="./aksharam_mt5",

        evaluation_strategy="no",

        learning_rate=1e-4,

        # 🔥 safe + efficient
        per_device_train_batch_size=6,
        gradient_accumulation_steps=2,

        num_train_epochs=1,

        # ❌ TURNED OFF (fixes NaN)
        fp16=False,

        logging_steps=100,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        remove_unused_columns=False,
        report_to="none"
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # -----------------------------
    # TRAIN
    # -----------------------------
    print("\n🚀 STABLE mT5 TRAINING STARTED...")
    trainer.train()

    # -----------------------------
    # SAVE
    # -----------------------------
    trainer.save_model("./aksharam_mt5_final")

    print("\n✅ TRAINING COMPLETE!")


if __name__ == "__main__":
    main()