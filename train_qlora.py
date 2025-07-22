# train_qlora.py
# Implements QLoRA (Quantized Low-Rank Adaptation) fine-tuning for the wheel loader action planning model.
# This script quantizes the base model to 4-bit precision and applies LoRA adapters for efficient training.

import os
import json
import torch
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

# Model and training configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Base instruction-tuned model
DATA_PATH = "final_trunc.jsonl"  # Path to preprocessed training data
OUTPUT_DIR = "qlora-qwen2.5-planner"  # Directory for saving model checkpoints
MAX_LENGTH = 384  # Maximum sequence length for tokenization
BATCH_SIZE = 16  # Training batch size
EPOCHS = 3  # Number of training epochs
LEARNING_RATE = 1e-4  # Initial learning rate
LORA_R = 8  # LoRA attention dimension
LORA_ALPHA = 16  # LoRA alpha parameter for scaling
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers

# Initialize tokenizer and configure 4-bit quantization
print("Loading tokenizer and quantized model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load base model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)

# Configure and apply LoRA adapters
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],  # Target attention projection matrices
    lora_dropout=LORA_DROPOUT,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Load and preprocess training dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": DATA_PATH}, split="train")

# System prompt template for instruction formatting
SYSTEM_PROMPT = (
    "System: You are a wheel-loader action planner. "
    "Given a JSON of detections and a command, generate a JSON plan."
)

def preprocess(examples):
    """
    Preprocesses training examples by formatting them with system prompt and tokenizing.
    Handles input/output pairs for causal language modeling.
    """
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
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=24  # Adjust based on available CPU cores
)

# Initialize data collator for causal language modeling
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,  # Keep only the last 2 checkpoints
    do_eval=False,
    remove_unused_columns=False,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer
)

# Execute training
print("Starting QLoRA fine-tuning...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
