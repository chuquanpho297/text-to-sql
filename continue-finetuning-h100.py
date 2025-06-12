#!/usr/bin/env python3

"""
Continue Fine-tuning Script for H100 GPU - XiYanSQL-QwenCoder-32B-2412

This script continues fine-tuning from a previously trained H100 model:
- Base model: hng229/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned-H100_79GB
- Optimized for H100 80GB GPU
- Continues training from existing checkpoint

Environment Variables Required:
- HUGGINGFACE_TOKEN: Your Hugging Face API token
- HUGGINGFACE_USERNAME: Your Hugging Face username

Hardware Requirements:
- Single H100 80GB GPU
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
import pandas as pd  # Used in data processing
from tqdm.auto import tqdm
import os
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    UnslothTrainingArguments,
)
import torch
from dotenv import load_dotenv
from mschema_implementation import sql_to_mschema
from datasets import Dataset, load_dataset, concatenate_datasets
from trl import SFTTrainer
from huggingface_hub import login, HfApi
import wandb

try:
    from unsloth.chat_templates import train_on_responses_only
except ImportError:
    print(
        "Warning: unsloth.chat_templates not available, will skip response-only training"
    )

# Load environment variables
load_dotenv()

# =============================================================================
# 3. Model Configuration & GPU Detection
# =============================================================================
# CONTINUE FROM EXISTING H100 MODEL
model_name = "hng229/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned-H100_79GB"
max_seq_length = 1024  # Keep same as original training
dtype = None
load_in_4bit = True  # Keep 4-bit quantization for consistency

# Fixed GPU configuration for H100
gpu_config = {
    "gpu_count": 1,
    "gpu_types": ["H100-80GB"],
    "total_memory": 80.0,
    "setup_type": "Single-H100-80GB",
    "individual_memory": [80.0]
}

print(f"🔍 Target GPU Setup: {gpu_config['setup_type']}")
print(f"📊 Expected GPU Count: {gpu_config['gpu_count']}")
print(f"💾 Expected GPU Memory: {gpu_config['total_memory']:.1f}GB")

# Verify actual GPU matches expected
if torch.cuda.is_available():
    actual_gpu_count = torch.cuda.device_count()
    if actual_gpu_count != 1:
        print(f"⚠️ Warning: Expected 1 GPU, found {actual_gpu_count}")
    
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    gpu_memory = props.total_memory / (1024**3)
    print(f"✅ Detected: {gpu_name} ({gpu_memory:.1f}GB)")
    
    if "H100" not in gpu_name:
        print(f"⚠️ Warning: Expected H100, found {gpu_name}")
else:
    raise ValueError("CUDA not available. H100 GPU required for this script.")

use_distributed = False

# =============================================================================
# 4. Initialize Weights & Biases with GPU Configuration
# =============================================================================
wandb_project = os.getenv("WANDB_PROJECT", "xiyanSQL-32b-continue-finetuning")
wandb_entity = os.getenv("WANDB_ENTITY", None)
wandb_run_name = f"xiyanSQL-32b-continue-H100-{max_seq_length}seq-200steps"

tags = ["sql", "text2sql", "32b", "lora", "unsloth", "continue-training", "h100"]

# H100 optimized settings
per_device_batch_size = 12
gradient_accumulation = 4
print("🚀 H100 80GB - Optimized for maximum performance (continue training)")

# Initialize wandb
wandb.init(
    project=wandb_project,
    name=wandb_run_name,
    config={
        # Model configuration
        "base_model": "hng229/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned-H100_79GB",
        "training_type": "continue_finetuning",
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "lora_rank": 16,
        "lora_alpha": 16,
        "dataset": "nguyenthetuyen/sql-text2sql-dataset",
        # Training configuration
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "effective_batch_size": per_device_batch_size * gradient_accumulation,
        "learning_rate": 4e-5,
        "max_steps": 200,
        "save_steps": 20,
        "logging_steps": 2,
        "optimizer": "adamw_torch_fused",
        "scheduler": "cosine_with_restarts",
        # Hardware configuration
        "gpu_setup_type": gpu_config["setup_type"],
        "total_gpu_memory_gb": gpu_config["total_memory"],
        "hardware_optimization": "H100_continue_optimized",
    },
    tags=tags,
    notes=f"Continue fine-tuning from H100 model: {model_name}",
)

print(f"🔗 Wandb tracking: {wandb.run.url}")

# =============================================================================
# 5. Load Existing Model for Continue Training
# =============================================================================
print(f"Loading existing model: {model_name}")
print("This model was previously fine-tuned on H100, continuing training...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

EOS_TOKEN = tokenizer.eos_token

# =============================================================================
# 6. LoRA Configuration - Same as Original
# =============================================================================
print("Applying LoRA configuration for continue training...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

# =============================================================================
# 7. Data preparation (SAME AS ORIGINAL)
# =============================================================================
ds = load_dataset("nguyenthetuyen/sql-text2sql-dataset")
print(ds)

merged_ds = concatenate_datasets([ds["train"], ds["val"]])
print(merged_ds)

def prepare_instruct(dataset):
    original_dataset = dataset
    df = original_dataset.to_pandas()
    tqdm.pandas()

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

    processed_dataset = Dataset.from_pandas(df)
    return processed_dataset

merged_ds = prepare_instruct(merged_ds)

system_prompt = """You are an AI assistant specialized in converting natural language questions into accurate SQL queries. Based on the question's context and an implicit database schema, generate an optimized, concise, and syntactically correct SQL query. Do not explain, only return the SQL."""

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

# =============================================================================
# 8. Continue Fine-tuning Configuration
# =============================================================================
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")

try:
    login(hf_token)
    print("✅ Successfully logged in to Hugging Face")
except Exception as login_error:
    raise ValueError(f"Failed to login to Hugging Face: {login_error}")

hf_username = os.getenv("HUGGINGFACE_USERNAME")
if not hf_username:
    raise ValueError("HUGGINGFACE_USERNAME not found in environment variables.")

# New model name for continued training
new_model_name = f"{hf_username}/XiYanSQL-QwenCoder-32B-2412-100kSQL_continued-H100_79GB"

print(f"🔍 Continue training will save to: {new_model_name}")

api = HfApi()
try:
    repo_url = api.create_repo(
        new_model_name,
        private=True,
        exist_ok=True,
        token=hf_token,
    )
    print(f"✅ Repository ready: {repo_url}")
except Exception as repo_error:
    print(f"⚠️ Warning: Could not verify repository access: {repo_error}")

print("CONTINUE TRAINING CONFIGURATION:")
print("- Continue from: H100 fine-tuned model")
print("- Additional steps: 200")
print("- Logging: Every 2 steps")
print("- Saving: Every 20 steps")
print(f"- Per-device batch size: {per_device_batch_size}")
print(f"- Gradient accumulation: {gradient_accumulation}")
print(f"- Effective batch size: {per_device_batch_size * gradient_accumulation}")

instruct_finetune_args = UnslothTrainingArguments(
    output_dir="/tmp/continue_finetune_h100",
    seed=3407,
    logging_steps=2,
    save_steps=20,
    save_strategy="steps",
    hub_strategy="end",
    push_to_hub=True,
    hub_model_id=new_model_name,
    hub_private_repo=True,
    # H100 OPTIMIZED SETTINGS:
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    warmup_steps=20,
    num_train_epochs=1.0,
    max_steps=200,
    fp16=False,
    bf16=True,
    learning_rate=2e-5,  # Lower LR for continue training
    embedding_learning_rate=1e-5,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine_with_restarts",
    lr_scheduler_kwargs={"num_cycles": 2},
    weight_decay=0.005,
    report_to="wandb",
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    remove_unused_columns=True,
    max_grad_norm=0.5,
    group_by_length=True,
    ddp_find_unused_parameters=False,
    save_safetensors=True,
    gradient_checkpointing=False,
    torch_compile=True,
    include_inputs_for_metrics=False,
    prediction_loss_only=True,
    save_only_model=True,
    load_best_model_at_end=False,
)

instruct_finetune_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=merged_ds,
    dataset_text_field="text",
    args=instruct_finetune_args,
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    packing=True,
    dataset_batch_size=1000,
)

# Apply response-only training if available
try:
    instruct_finetune_trainer = train_on_responses_only(
        instruct_finetune_trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    print("✓ Applied response-only training optimization")
except NameError:
    print("⚠ Skipping response-only training (chat_templates not available)")
    pass

# =============================================================================
# 9. Training Execution
# =============================================================================
def print_memory_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")

if __name__ == "__main__":
    print("🚀 CONTINUE FINE-TUNING FROM H100 MODEL")
    print("=" * 60)
    print(f"📦 Base Model: {model_name}")
    print(f"🎯 New Model: {new_model_name}")
    print("🔧 Continue training with lower learning rate")
    print("💾 H100 80GB optimization enabled")
    print("=" * 60)

    print("Initial memory usage:")
    print_memory_usage()

    try:
        instruct_finetune_trainer_stats = instruct_finetune_trainer.train()
        print("Continue fine-tuning completed successfully!")
        print(f"🔗 Model uploaded to: https://huggingface.co/{new_model_name}")

        print("Final memory usage:")
        print_memory_usage()

    except torch.cuda.OutOfMemoryError as e:
        print("=" * 60)
        print("OUT OF MEMORY ERROR!")
        print("=" * 60)
        print("Try reducing batch size or sequence length")
        raise e
    except Exception as e:
        print(f"Training failed with error: {e}")
        print_memory_usage()
        raise e
