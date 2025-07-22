# train_qlora.py
# QLoRA fine-tuning script for end-to-end loader planning model

import os
import json
import torch
import signal
import sys
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# === Configuration ===
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # adjust if needed
DATA_PATH = "final_trunc.jsonl"
OUTPUT_DIR = "qlora-qwen2.5-planner"
MAX_LENGTH = 384
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# === Load tokenizer and model (4-bit quantized) ===
print("Loading tokenizer and quantized model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)

# === Apply LoRA adapters ===
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none"
)
model = get_peft_model(model, lora_config)

# === Load and preprocess dataset ===
print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")

# Format examples for causal LM

SYSTEM_PROMPT = (
    "System: You are a wheel-loader action planner. "
    "Given a JSON of detections and a command, generate a JSON plan."
)

def preprocess(examples):
    inputs = []
    for inp, out in zip(examples["input"], examples["output"]):
        inp_str = json.dumps(inp, ensure_ascii=False)
        out_str = json.dumps(out, ensure_ascii=False)
        text = f"{SYSTEM_PROMPT}\nInput: {inp_str}\nOutput: {out_str}"
        inputs.append(text)
    tokenized = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing...")
dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=24
)

# === Data collator ===
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Training setup ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    do_eval=False,  # Changed from evaluation_strategy
    remove_unused_columns=False,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer
)

# === Start training ===
print("Starting QLoRA fine-tuning...")

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Saving model before exiting...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
