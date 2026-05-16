import os
import json
import torch
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType


def load_data(path):
    print("Loading dataset...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Total samples:", len(data))
    return Dataset.from_list(data)


def preprocess(examples, tokenizer):
    return tokenizer(
        examples["en"],
        text_target=examples["ml"],
        max_length=128,
        truncation=True
    )


def main():

    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    data_path = "final_dataset.json"

    if not os.path.exists(data_path):
        print("Dataset not found!")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "ml_IN"

    # Dataset
    dataset = load_data(data_path)
    dataset = dataset.train_test_split(test_size=0.1)

    tokenized = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Model
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.to(device)

    model.print_trainable_parameters()

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="./aksharam_model",

        learning_rate=2e-4,

        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,

        num_train_epochs=2,

        fp16=torch.cuda.is_available(),

        save_steps=1000,
        save_total_limit=2,

        logging_steps=50,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print("🚀 Training started...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("./aksharam_model_final")

    print("✅ Done!")


if __name__ == "__main__":
    main()