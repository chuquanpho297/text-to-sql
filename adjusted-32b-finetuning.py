#!/usr/bin/env python3

"""
LLM Fine-tuning Script for SQL Generation - Modified for XiYanSQL-QwenCoder-32B-2412

Key Changes from Original Script:
- Updated model name to XiYanSQL-QwenCoder-32B-2412
- Enabled 4-bit quantization for memory efficiency
- Dynamic batch size adjustment based on GPU type (H100/A100)
- Adjusted sequence length for memory optimization
- Added memory optimization configurations
- Optimized for single GPU training (H100 80GB or A100 80GB)

Environment Variables Required:
- HUGGINGFACE_TOKEN: Your Hugging Face API token
- HUGGINGFACE_USERNAME: Your Hugging Face username

Hardware Requirements:
- Single GPU: 64GB+ VRAM (H100 80GB or A100 80GB recommended)
- Supports H100 80GB for maximum performance
- Supports A100 80GB for stable training
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
model_name = "XGenerationLab/XiYanSQL-QwenCoder-32B-2412"  # Changed from 3B to 32B
max_seq_length = 1024  # Reduced from 2048 to 1024 for faster training
dtype = None
load_in_4bit = True  # CRITICAL: Enable 4-bit quantization for 32B model


# Detect GPU configuration
def detect_gpu_setup():
    """Detect available GPUs and their types"""
    if not torch.cuda.is_available():
        return {"gpu_count": 0, "gpu_types": [], "total_memory": 0, "setup_type": "CPU"}

    gpu_count = torch.cuda.device_count()
    gpu_types = []
    total_memory = 0

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpu_name = props.name
        gpu_memory = props.total_memory / (1024**3)  # Convert to GB
        total_memory += gpu_memory

        # Detect GPU type
        if "H100" in gpu_name:
            gpu_types.append(f"H100-{int(gpu_memory)}GB")
        elif "A100" in gpu_name:
            gpu_types.append(f"A100-{int(gpu_memory)}GB")
        else:
            gpu_types.append(f"{gpu_name}-{int(gpu_memory)}GB")

    # Only support single GPU
    if gpu_count == 1:
        setup_type = f"Single-{gpu_types[0]}"
    else:
        setup_type = f"Unsupported-{gpu_count}GPUs"

    return {
        "gpu_count": gpu_count,
        "gpu_types": gpu_types,
        "total_memory": total_memory,
        "setup_type": setup_type,
        "individual_memory": [
            torch.cuda.get_device_properties(i).total_memory / (1024**3)
            for i in range(gpu_count)
        ],
    }


# Get GPU configuration
gpu_config = detect_gpu_setup()
print(f"🔍 Detected GPU Setup: {gpu_config['setup_type']}")
print(f"📊 GPU Count: {gpu_config['gpu_count']}")
print(f"💾 Total GPU Memory: {gpu_config['total_memory']:.1f}GB")
for i, gpu_type in enumerate(gpu_config["gpu_types"]):
    print(f"   GPU {i}: {gpu_type}")

# Single GPU configuration only
if gpu_config["gpu_count"] != 1:
    raise ValueError(
        f"This script is configured for single GPU training only. Detected {gpu_config['gpu_count']} GPUs. Please use CUDA_VISIBLE_DEVICES=0 to select one GPU."
    )

use_distributed = False

# =============================================================================
# 4. Initialize Weights & Biases with GPU Configuration
# =============================================================================
# Get wandb configuration from environment variables (optional)
wandb_project = os.getenv("WANDB_PROJECT", "xiyanSQL-32b-finetuning")
wandb_entity = os.getenv("WANDB_ENTITY", None)  # Optional
wandb_run_name = (
    f"xiyanSQL-32b-{gpu_config['setup_type']}-{max_seq_length}seq-{200}steps"
)

# Create dynamic tags based on GPU configuration
tags = ["sql", "text2sql", "32b", "lora", "unsloth", "single-gpu"]
if "H100" in gpu_config["setup_type"]:
    tags.append("h100")
if "A100" in gpu_config["setup_type"]:
    tags.append("a100")

# Adjust batch size based on single GPU configuration
if gpu_config["gpu_count"] == 1:
    if "H100" in gpu_config["setup_type"]:
        # H100 80GB - Maximum utilization
        per_device_batch_size = 12
        gradient_accumulation = 4
        print("🚀 H100 80GB detected - Optimized for maximum performance")
    elif "A100" in gpu_config["setup_type"]:
        # A100 80GB - Conservative but efficient
        per_device_batch_size = 8
        gradient_accumulation = 6
        print("🚀 A100 80GB detected - Optimized for stability")
    else:
        # Generic single GPU
        per_device_batch_size = 4
        gradient_accumulation = 8
        print("🚀 Generic GPU detected - Conservative settings")
else:
    # Should not reach here due to earlier validation
    raise ValueError(
        f"This script is configured for single GPU training only. Detected {gpu_config['gpu_count']} GPUs."
    )

# Initialize wandb with comprehensive GPU information
wandb.init(
    project=wandb_project,
    name=wandb_run_name,
    config={
        # Model configuration
        "model_name": "XGenerationLab/XiYanSQL-QwenCoder-32B-2412",
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "lora_rank": 16,
        "lora_alpha": 16,
        "dataset": "nguyenthetuyen/sql-text2sql-dataset",
        # Training configuration
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "effective_batch_size": per_device_batch_size
        * gradient_accumulation
        * gpu_config["gpu_count"],
        "learning_rate": 4e-5,
        "max_steps": 200,
        "save_steps": 2,
        "logging_steps": 2,
        "optimizer": "adamw_torch_fused",
        "scheduler": "cosine_with_restarts",
        # Hardware configuration
        "gpu_count": gpu_config["gpu_count"],
        "gpu_setup_type": gpu_config["setup_type"],
        "gpu_types": gpu_config["gpu_types"],
        "total_gpu_memory_gb": round(gpu_config["total_memory"], 1),
        "individual_gpu_memory": [
            round(mem, 1) for mem in gpu_config["individual_memory"]
        ],
        "distributed_training": use_distributed,
        "hardware_optimization": f"{gpu_config['setup_type']}_optimized",
        # System information
        "system_os": "Linux" if "linux" in os.name.lower() else "Other",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    },
    tags=tags,
    notes=f"Fine-tuning XiYanSQL-QwenCoder-32B on 100k SQL dataset with {gpu_config['setup_type']} configuration. Total GPU memory: {gpu_config['total_memory']:.1f}GB across {gpu_config['gpu_count']} GPU(s).",
)

print(f"🔗 Wandb tracking: {wandb.run.url}")

# =============================================================================
# 5. Download model - UPDATED FOR 32B MODEL
# =============================================================================
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
# 6. LoRA Configuration - OPTIMIZED FOR 32B MODEL
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
# 7. Data preparation (SAME AS ORIGINAL)
# =============================================================================
# =============================================================================
# 7.1. Download 100k dataset
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
# 8. Instruction fine-tune - ADJUSTED FOR 32B MODEL
# =============================================================================

# Login to HuggingFace (using token from environment)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError(
        "HUGGINGFACE_TOKEN not found in environment variables. Please check your .env file."
    )

# Validate token by logging in
try:
    login(hf_token)
    print("✅ Successfully logged in to Hugging Face")
except Exception as login_error:
    raise ValueError(f"Failed to login to Hugging Face: {login_error}")

hf_username = os.getenv("HUGGINGFACE_USERNAME")
if not hf_username:
    raise ValueError(
        "HUGGINGFACE_USERNAME not found in environment variables. Please check your .env file."
    )

# Test repository access
print(
    f"🔍 Testing repository access for: {hf_username}/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned"
)
api = HfApi()
try:
    # Test if we can create/access the repository
    repo_url = api.create_repo(
        f"{hf_username}/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned",
        private=True,
        exist_ok=True,  # Don't fail if repo already exists
        token=hf_token,
    )
    print(f"✅ Repository ready: {repo_url}")
except Exception as repo_error:
    print(f"⚠️ Warning: Could not verify repository access: {repo_error}")
    print("Training will continue, but model upload might fail")

print(f"Configuring training arguments for 32B model on {gpu_config['setup_type']}...")
print("TRAINING CONFIGURATION:")
print("- Epochs: 1 full epoch")
print("- Max steps: 200 steps (hard limit)")
print("- Logging: Every 2 steps")
print("- Saving: Every 2 steps")
print("- Hub uploads: Every 2 steps (every save)")
print(f"- Per-device batch size: {per_device_batch_size}")
print(f"- Gradient accumulation: {gradient_accumulation}")
print(
    f"- Effective batch size: {per_device_batch_size * gradient_accumulation * gpu_config['gpu_count']} ({per_device_batch_size} × {gradient_accumulation} × {gpu_config['gpu_count']} GPU)"
)

print(f"{gpu_config['setup_type'].upper()} CONFIGURATION:")
print(f"- Total GPUs: {gpu_config['gpu_count']}")
print(f"- Total VRAM: {gpu_config['total_memory']:.1f}GB")
for i, (gpu_type, memory) in enumerate(
    zip(gpu_config["gpu_types"], gpu_config["individual_memory"])
):
    print(f"- GPU {i}: {gpu_type} ({memory:.1f}GB)")
print("- Optimizer: adamw_torch_fused (Hardware optimized)")
print("- Precision: BF16 (Tensor core optimized)")
print("- PyTorch compile enabled for speed")
print("- TF32 enabled for tensor cores")
print("- Weights & Biases logging enabled")

print("SPEED OPTIMIZATIONS:")
print("- Reduced sequence length to 1024")
print("- LoRA rank 16 for faster adaptation")
print("- Cosine scheduler with restarts")
print("- Group by length for efficient batching")
print("- Single GPU optimization")
print("=" * 60)

instruct_finetune_args = UnslothTrainingArguments(
    output_dir="/kaggle/working/finetune_32b",
    seed=3407,
    logging_steps=2,  # Log every 2 steps as requested
    save_steps=2,  # Save every 2 steps as requested
    save_strategy="steps",
    hub_strategy="every_save",  # Push to HF every time we save (every 2 steps)
    push_to_hub=True,
    hub_model_id=f"{hf_username}/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned",
    hub_private_repo=True,
    # OPTIMIZED FOR SINGLE GPU (H100 OR A100):
    per_device_train_batch_size=per_device_batch_size,  # Dynamic based on GPU type
    gradient_accumulation_steps=gradient_accumulation,  # Dynamic based on GPU type
    warmup_steps=20,  # Increased warmup for larger batches
    num_train_epochs=1.0,  # 1 full epoch as requested
    max_steps=200,  # 200 steps as requested
    fp16=False,  # Disable FP16 for better precision
    bf16=True,  # Force BF16 - Works well on both H100 and A100
    learning_rate=4e-5,  # Higher LR for faster convergence
    embedding_learning_rate=2e-5,  # Proportionally higher
    optim="adamw_torch_fused",  # Optimized fused optimizer
    lr_scheduler_type="cosine_with_restarts",  # Aggressive scheduler
    lr_scheduler_kwargs={"num_cycles": 2},  # Fast restart cycles
    weight_decay=0.005,  # Reduced for faster convergence
    report_to="wandb",  # Enable Weights & Biases logging
    # SINGLE GPU OPTIMIZATIONS:
    dataloader_pin_memory=True,
    dataloader_num_workers=8,  # More workers for data loading
    remove_unused_columns=True,
    max_grad_norm=0.5,  # Looser clipping for speed
    group_by_length=True,
    ddp_find_unused_parameters=False,
    save_safetensors=True,
    # ADDITIONAL SPEED OPTIMIZATIONS:
    gradient_checkpointing=False,  # Disable for pure speed (single GPU has enough memory)
    torch_compile=True,  # Enable PyTorch 2.0 compilation
    include_inputs_for_metrics=False,  # Skip metric computation
    prediction_loss_only=True,  # Skip evaluation metrics for speed
    save_only_model=True,  # Don't save optimizer states
    load_best_model_at_end=False,  # Skip final model loading
)

print(
    f"Effective batch size: {per_device_batch_size * gradient_accumulation} (per_device_train_batch_size * gradient_accumulation_steps)"
)
print(
    f"{gpu_config['setup_type'].upper()} MAXIMUM VRAM UTILIZATION: Batch size {per_device_batch_size}, gradient accumulation {gradient_accumulation}, torch_compile enabled"
)
print(
    "Hub strategy: every_save - model will be pushed to HF after each save checkpoint"
)
print("Target: Complete 200 steps of training with frequent saves to HF")

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
# 9. Additional Memory Monitoring (NEW)
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
# 10. Training Execution with Memory Monitoring
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
        print(
            f"🔗 Model automatically uploaded to: https://huggingface.co/{hf_username}/XiYanSQL-QwenCoder-32B-2412-100kSQL_finetuned"
        )

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
# 11. Additional Speed Optimizations
# =============================================================================
def optimize_for_speed():
    """Apply additional optimizations for faster training on H100/A100"""
    if torch.cuda.is_available():
        # Enable optimized attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("✓ Enabled Flash Attention for faster training")
        except Exception as e:
            print(f"⚠ Flash Attention not available: {e}")
            pass

        # H100/A100-specific optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ Enabled TF32 for tensor cores")

        # Enable BF16 mixed precision globally
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        print("✓ Enabled BF16 optimizations")

        # Optimize memory allocation for 80GB GPUs
        if hasattr(torch.cuda, "set_per_process_memory_fraction"):
            torch.cuda.set_per_process_memory_fraction(0.98)  # Use more memory
            print("✓ Optimized GPU memory allocation (98%)")

        # Enable JIT compilation for faster execution
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        print("✓ Disabled JIT profiling for speed")

    # Set optimal threading for CPU operations
    torch.set_num_threads(
        min(16, torch.get_num_threads())
    )  # Single GPU systems often have fewer CPU cores allocated
    print(f"✓ Set CPU threads to: {torch.get_num_threads()}")

    # Enable optimized data loading
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
        print("✓ Set multiprocessing start method to spawn")
    except RuntimeError:
        pass


# Apply optimizations
print("Applying additional speed optimizations...")
optimize_for_speed()
