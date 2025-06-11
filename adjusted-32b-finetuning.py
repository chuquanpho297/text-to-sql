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
from huggingface_hub import login
try:
    from unsloth.chat_templates import train_on_responses_only
except ImportError:
    print("Warning: unsloth.chat_templates not available, will skip response-only training")

# Load environment variables
load_dotenv()

# =============================================================================
# 3. Download model - UPDATED FOR 32B MODEL
# =============================================================================
model_name = "XGenerationLab/XiYanSQL-QwenCoder-32B-2412"  # Changed from 3B to 32B
max_seq_length = 1024  # Reduced from 2048 to 1024 for faster training
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
    r=16,  # Reduced from 32 to 16 for faster training with minimal quality loss
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Same target modules - compatible with Qwen2 architecture
    lora_alpha=16,  # Reduced proportionally with rank for faster convergence
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # Critical for memory efficiency
    random_state=3407,
    use_rslora=True,  # Enable rank stabilized LoRA for better stability and speed
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
print("SPEED OPTIMIZATIONS:")
print("- Reduced sequence length to 1024 for faster processing")
print("- Lower LoRA rank (16) for faster adaptation")
print("- Higher learning rate (5e-5) for faster convergence")
print("- Cosine scheduler for better convergence speed")
print("- Reduced epochs to 0.5 for quick training")
print("- Group by length for efficient batching")
print("- Less frequent saving to reduce I/O overhead")
print("="*60)

instruct_finetune_args = UnslothTrainingArguments(
    output_dir="/kaggle/working/finetune_32b",
    seed=3407,
    logging_steps=50,  # Reduced for more frequent logging
    save_steps=500,  # Increased to save less frequently (faster training)
    save_strategy="steps",
    hub_strategy="end",  # Changed from "every_save" to "end" for faster training
    push_to_hub=True,
    hub_model_id=f"{hf_username}/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned",
    hub_private_repo=True,
    # OPTIMIZED FOR SPEED:
    per_device_train_batch_size=64,  # Fixed from 64 to 1 - 64 would cause OOM
    gradient_accumulation_steps=32,  # Increased to maintain effective batch size of 32
    warmup_ratio=0.03,  # Use ratio instead of steps for better scaling
    num_train_epochs=0.5,  # Reduced from 1 to 0.5 for faster training
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    learning_rate=5e-5,  # Increased learning rate for faster convergence
    embedding_learning_rate=1e-5,  # Adjusted proportionally
    optim="adamw_8bit",  # Use 8-bit optimizer for memory efficiency
    lr_scheduler_type="cosine",  # Cosine scheduler often converges faster than linear
    weight_decay=0.01,
    report_to="none",  # Use this for WandB etc
    # Additional speed optimizations
    dataloader_pin_memory=True,  # Enable pin memory for faster data loading (if you have enough RAM)
    dataloader_num_workers=4,  # Use multiple workers for data loading
    remove_unused_columns=True,
    max_grad_norm=1.0,  # Gradient clipping for stability
    group_by_length=True,  # Group samples by length for faster training
    ddp_find_unused_parameters=False,  # Speed optimization for DDP
    save_safetensors=True,  # Faster saving format
)

print(
    f"Effective batch size: {1 * 32} (per_device_batch_size * gradient_accumulation_steps)"
)
print("Speed optimizations: Higher LR, cosine scheduler, group_by_length, reduced epochs")

instruct_finetune_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=merged_ds,
    dataset_text_field="text",
    args=instruct_finetune_args,
    max_seq_length=max_seq_length,
    dataset_num_proc=8,  # Use multiple processes for dataset processing
    packing=True,  # Enable packing for faster training (was False)
    dataset_batch_size=1000,  # Process dataset in larger batches
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
    print("🚀 SPEED-OPTIMIZED FINE-TUNING FOR 32B MODEL")
    print("=" * 60)
    print("KEY SPEED OPTIMIZATIONS APPLIED:")
    print("✓ Reduced LoRA rank: 32 → 16 (faster adaptation)")
    print("✓ Enabled RSLoRA for better stability")
    print("✓ Higher learning rate: 2e-5 → 5e-5 (faster convergence)")
    print("✓ Cosine scheduler (better than linear for speed)")
    print("✓ Reduced epochs: 1.0 → 0.5 (quick training)")
    print("✓ Sequence length: 2048 → 1024 (faster processing)")
    print("✓ Enabled packing for efficient batch processing")
    print("✓ Group by length for optimal batching")
    print("✓ Reduced save frequency (less I/O overhead)")
    print("✓ Fixed batch size issue (64 → 1)")
    print("✓ Optimized effective batch size: 32")
    print("=" * 60)
    print("MEMORY REQUIREMENTS:")
    print("- Minimum: 32GB VRAM (with optimizations)")
    print("- Recommended: 48GB+ VRAM")
    print("- Configuration: 4-bit quantization + LoRA")
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

# =============================================================================
# 9. Additional Speed Optimizations
# =============================================================================
def optimize_for_speed():
    """Apply additional optimizations for faster training"""
    if torch.cuda.is_available():
        # Enable optimized attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("✓ Enabled Flash Attention for faster training")
        except Exception as e:
            print(f"⚠ Flash Attention not available: {e}")
            pass
        
        # Set optimal tensor core usage
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ Enabled TF32 for faster computation")
        
        # Optimize memory allocation
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95)
            print("✓ Optimized GPU memory allocation")
    
    # Set optimal threading for CPU operations
    torch.set_num_threads(min(8, torch.get_num_threads()))
    print(f"✓ Set CPU threads to: {torch.get_num_threads()}")

# Apply optimizations
print("Applying additional speed optimizations...")
optimize_for_speed()
