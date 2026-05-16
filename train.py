
import os
import json
import torch
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType


def load_aligned_data(file_path):

    print(f"Loading data from {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    return Dataset.from_list(data)


def preprocess_function(examples, tokenizer, max_length=256):

    inputs = examples["en"]
    targets = examples["ml"]

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=max_length,
        truncation=True
    )

    return model_inputs


def main():

    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    data_path = "aligned_dataset.json"

    if not os.path.exists(data_path):
        print("Dataset not found. Run collect script first.")
        return


    # -----------------------------
    # Tokenizer
    # -----------------------------
    print("Loading tokenizer...")

    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "ml_IN"


    # -----------------------------
    # Dataset
    # -----------------------------
    print("Loading dataset...")

    dataset = load_aligned_data(data_path)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)


    # -----------------------------
    # Tokenization
    # -----------------------------
    print("Tokenizing dataset...")

    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )


    # -----------------------------
    # Model
    # -----------------------------
    print("Loading model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model.to(device)


    # -----------------------------
    # Apply LoRA
    # -----------------------------
    print("Applying LoRA...")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()


    # -----------------------------
    # Data collator
    # -----------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model
    )


    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = Seq2SeqTrainingArguments(

        output_dir="./aksharam_model",

        learning_rate=2e-4,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        weight_decay=0.01,

        save_total_limit=2,

        num_train_epochs=6,

        predict_with_generate=True,

        fp16=torch.cuda.is_available(),

        logging_steps=100,

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

    eval_dataset=tokenized_datasets["test"],

    data_collator=data_collator
    )
    


    # -----------------------------
    # Train
    # -----------------------------
    print("Starting training...")

    trainer.train()


    # -----------------------------
    # Save model
    # -----------------------------
    print("Saving model...")

    trainer.save_model("./aksharam_model_final")

    print("Training complete!")

    print("Model saved to ./aksharam_model_final")


if __name__ == "__main__":
    main()
