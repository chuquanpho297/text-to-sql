#!/usr/bin/env python3

"""
LLM Fine-tuning Script for SQL Generation - Modified for XiYanSQL-QwenCoder-32B-2412

Key Changes from Original Script:
- Updated model name to XiYanSQL-QwenCoder-32B-2412
- Enabled 4-bit quantization for memory efficiency
- Reduced batch size and increased gradient accumulation
- Adjusted sequence length for memory optimization
- Added memory optimization configurations

Environment Variables Required:
- HUGGINGFACE_TOKEN: Your Hugging Face API token
- HUGGINGFACE_USERNAME: Your Hugging Face username

Hardware Requirements:
- Minimum: 64GB VRAM (A100 80GB recommended)
- Optimal: 80GB+ VRAM for comfortable training
"""

# =============================================================================
# 1. Package Installation (run these commands manually if needed)
# =============================================================================
# pip install unsloth
# pip install huggingface_hub
# pip install bitsandbytes

# =============================================================================
# 2. Import modules
# =============================================================================
import re
import pandas as pd
from tqdm.auto import tqdm
import os
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    UnslothTrainer,
    UnslothTrainingArguments,
)
import torch
import json
from dotenv import load_dotenv
from mschema_implementation import sql_to_mschema
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login

# Load environment variables
load_dotenv()

# =============================================================================
# 3. Download model - UPDATED FOR 32B MODEL
# =============================================================================
model_name = "XGenerationLab/XiYanSQL-QwenCoder-32B-2412"  # Changed from 3B to 32B
max_seq_length = 2048  # Reduced from 1024 to 2048 for memory efficiency
dtype = None
load_in_4bit = True  # CRITICAL: Enable 4-bit quantization for 32B model

# WARNING: Ensure you have adequate VRAM (64GB+ recommended)
print(f"Loading {model_name} with 4-bit quantization...")
print(f"Sequence length: {max_seq_length}")
print("Hardware requirement: 64GB+ VRAM")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# get EOS_TOKEN to add to model input
EOS_TOKEN = tokenizer.eos_token

# =============================================================================
# 4. LoRA Configuration - OPTIMIZED FOR 32B MODEL
# =============================================================================
print("Applying LoRA configuration optimized for 32B model...")

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Increased from 16 to 32 for better adaptation on larger model
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Same target modules - compatible with Qwen2 architecture
    lora_alpha=32,  # Increased proportionally with rank
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # Critical for memory efficiency
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# =============================================================================
# 5. Data preparation (SAME AS ORIGINAL)
# =============================================================================
# 5.1. Download 100k dataset
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nguyenthetuyen/sql-text2sql-dataset")
print(ds)

# Merge two dataset splits
merged_ds = concatenate_datasets([ds["train"], ds["val"]])
print(merged_ds)

# Display a sample datapoint
for key, value in merged_ds[0].items():
    print(f"{key}: {value}")


# Defines input and output of the instruction fine-tuning dataset
def prepare_instruct(dataset):
    # Giả sử bạn đã tải dataset ban đầu
    original_dataset = dataset
    # Chuyển đổi dataset thành DataFrame để xử lý
    df = original_dataset.to_pandas()
    # Khởi tạo tqdm cho pandas
    tqdm.pandas()

    # Tạo cột input bằng cách nối sql_prompt với sql_context
    # Sử dụng sql_to_mschema thay vì sql_to_table và lấy db_name từ row["domain"]
    df["prompt"] = df.progress_apply(
        lambda row: """You are now a SQL data analyst, and you are given a database schema as follows:

【Schema】
{db_schema}

【Question】
{question}

Please read and understand the database schema carefully, and generate an executable SQL based on the user's question. The generated SQL is protected by ```sql and ```.
""".format(
            db_schema=sql_to_mschema(
                row["sql_context"], row.get("domain", "database")
            ).strip(),
            question=row["sql_prompt"].strip(),
        ),
        axis=1,
    )

    # Chuyển đổi lại thành dataset của Hugging Face
    processed_dataset = Dataset.from_pandas(df)
    return processed_dataset


# Apply prepare_instruct func to instruct_ds dataset
merged_ds = prepare_instruct(merged_ds)

# Display something to check if thing's ok
print(merged_ds)

system_prompt = """You are an AI assistant specialized in converting natural language questions into accurate SQL queries. Based on the question's context and an implicit database schema, generate an optimized, concise, and syntactically correct SQL query. Do not explain, only return the SQL."""

print(merged_ds[0]["prompt"] + "\n\nAnswer:\n" + merged_ds[0]["sql"])


def formatting_prompts_func(example):
    prompt = example["prompt"]
    sql = example["sql"]
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": sql},
    ]
    text = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


merged_ds = merged_ds.map(formatting_prompts_func, num_proc=4)
print(merged_ds["text"][:5])

# =============================================================================
# 6. Instruction fine-tune - ADJUSTED FOR 32B MODEL
# =============================================================================

# Login to HuggingFace (using token from environment)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError(
        "HUGGINGFACE_TOKEN not found in environment variables. Please check your .env file."
    )

login(hf_token)

hf_username = os.getenv("HUGGINGFACE_USERNAME")
if not hf_username:
    raise ValueError(
        "HUGGINGFACE_USERNAME not found in environment variables. Please check your .env file."
    )

print("Configuring training arguments for 32B model...")
print(
    "CRITICAL: Reduced batch size and increased gradient accumulation for memory efficiency"
)

instruct_finetune_args = UnslothTrainingArguments(
    output_dir="/kaggle/working/finetune_32b",
    seed=3407,
    logging_steps=200,
    save_steps=200,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=f"{hf_username}/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned",
    hub_private_repo=True,
    # CRITICAL CHANGES FOR 32B MODEL:
    per_device_train_batch_size=64,  # Reduced from 4 to 1 for memory
    gradient_accumulation_steps=16,  # Increased from 4 to 16 to maintain effective batch size of 16
    warmup_steps=5,
    num_train_epochs=1,  # Set this for 1 full training run.
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    learning_rate=2e-5,
    embedding_learning_rate=2e-6,
    optim="adamw_8bit",  # Use 8-bit optimizer for memory efficiency
    lr_scheduler_type="linear",
    weight_decay=0.01,
    report_to="none",  # Use this for WandB etc
    # Additional memory optimization
    dataloader_pin_memory=False,  # Disable pin memory to save RAM
    remove_unused_columns=True,
    max_grad_norm=1.0,  # Gradient clipping for stability
)

print(
    f"Effective batch size: {1 * 16} (per_device_batch_size * gradient_accumulation_steps)"
)
print("Memory optimization: 8-bit optimizer, gradient checkpointing enabled")

instruct_finetune_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=merged_ds,
    dataset_text_field="text",
    args=instruct_finetune_args,
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    packing=False,
)

from unsloth.chat_templates import train_on_responses_only

instruct_finetune_trainer = train_on_responses_only(
    instruct_finetune_trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)


# =============================================================================
# 7. Additional Memory Monitoring (NEW)
# =============================================================================
def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # Convert to GB
            print(
                f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved"
            )


# =============================================================================
# 8. Training Execution with Memory Monitoring
# =============================================================================
if __name__ == "__main__":
    print("Starting fine-tuning for 32B model...")
    print("=" * 60)
    print("MEMORY REQUIREMENTS:")
    print("- Minimum: 64GB VRAM")
    print("- Recommended: 80GB+ VRAM")
    print("- This configuration uses 4-bit quantization + LoRA for efficiency")
    print("=" * 60)

    # Print initial memory usage
    print("Initial memory usage:")
    print_memory_usage()

    try:
        instruct_finetune_trainer_stats = instruct_finetune_trainer.train()
        print("Fine-tuning completed successfully!")

        # Print final memory usage
        print("Final memory usage:")
        print_memory_usage()

    except torch.cuda.OutOfMemoryError as e:
        print("=" * 60)
        print("OUT OF MEMORY ERROR!")
        print("=" * 60)
        print("Suggestions:")
        print("1. Reduce per_device_train_batch_size to 1 (if not already)")
        print("2. Reduce max_seq_length further (try 1024)")
        print("3. Increase gradient_accumulation_steps")
        print("4. Ensure you have at least 64GB VRAM")
        print("5. Consider using DeepSpeed ZeRO for multi-GPU setup")
        print("=" * 60)
        raise e
    except Exception as e:
        print(f"Training failed with error: {e}")
        print_memory_usage()
        raise e
