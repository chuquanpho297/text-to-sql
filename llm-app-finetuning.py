#!/usr/bin/env python3
"""
LLM Fine-tuning Script for SQL Generation
Converted from Jupyter notebook: llm-app-thamkhao.ipynb
Excludes test section as requested

Environment Variables Required:
- HUGGINGFACE_TOKEN: Your Hugging Face API token
- HUGGINGFACE_USERNAME: Your Hugging Face username
Copy .env.example to .env and fill in your values.
"""

# =============================================================================
# 1. Package Installation (run these commands manually if needed)
# =============================================================================
# pip install unsloth
# pip install huggingface_hub

# =============================================================================
# 2. Import modules
# =============================================================================
import re
import pandas as pd
from tqdm.auto import tqdm
import os
import torch
import json
from dotenv import load_dotenv

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    UnslothTrainer,
    UnslothTrainingArguments,
)
from trl import SFTTrainer

from huggingface_hub import login

# Load environment variables
load_dotenv()

# =============================================================================
# 3. Download model
# =============================================================================
model_name = "XGenerationLab/XiYanSQL-QwenCoder-3B-2502"
max_seq_length = 1024
dtype = None
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# get EOS_TOKEN to add to model input
EOS_TOKEN = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# =============================================================================
# 4. Data preparation
# =============================================================================

# 4.1. Download 100k dataset
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nguyenthetuyen/sql-text2sql-dataset")
print(ds)

# Merge two dataset splits
merged_ds = concatenate_datasets([ds["train"], ds["val"]])
print(merged_ds)

# Display a sample datapoint
for key, value in merged_ds[0].items():
    print(f"{key}: {value}")


# 4.3. Convert SQL script to CSV table
def sql_to_table(context):
    sql_script = context

    # Biểu thức chính quy để trích xuất CREATE TABLE và INSERT INTO
    create_pattern = re.compile(r"CREATE TABLE (\w+) \((.*?)\);", re.DOTALL)
    insert_pattern = re.compile(r"INSERT INTO (\w+) \((.*?)\) VALUES (.*?);", re.DOTALL)

    table_schemas = {}  # Lưu schema của từng bảng
    table_data = {}  # Lưu dữ liệu từng bảng dưới dạng danh sách dict

    # Xử lý CREATE TABLE để lấy schema
    for match in create_pattern.finditer(sql_script):
        table_name = match.group(1)
        columns = [
            col.split()[0].strip() for col in match.group(2).split(",")
        ]  # Lấy tên cột, bỏ kiểu dữ liệu
        table_schemas[table_name] = columns
        table_data[table_name] = []  # Khởi tạo danh sách rỗng cho bảng này

    # Xử lý INSERT INTO để lấy dữ liệu
    for match in insert_pattern.finditer(sql_script):
        table_name = match.group(1)
        if table_name not in table_schemas:
            continue  # Bỏ qua nếu không có schema

        columns = [
            col.strip() for col in match.group(2).split(",")
        ]  # Các cột có dữ liệu
        values = match.group(3)

        # Chuyển đổi dữ liệu VALUES thành danh sách
        rows = re.findall(r"\((.*?)\)", values)
        for row in rows:
            row_values = [
                value.strip() for value in row.split(",")
            ]  # Chuyển đổi giá trị
            row_dict = {
                col: "NULL" for col in table_schemas[table_name]
            }  # Mặc định NULL cho tất cả cột

            # Gán giá trị theo thứ tự xuất hiện trong INSERT
            for col, val in zip(columns, row_values):
                row_dict[col] = val

            table_data[table_name].append(row_dict)

    # Chuyển đổi dữ liệu thành DataFrame và xuất CSV dạng chuỗi
    table_strings = []
    for table_name, rows in table_data.items():
        df = pd.DataFrame(
            rows, columns=table_schemas[table_name]
        )  # Giữ đúng thứ tự cột
        # df = df.fillna("NULL")  # Thay thế các giá trị None bằng NULL
        table_csv = df.to_csv(index=False)
        table_strings.append(f"Table: {table_name}\n{table_csv}")

    return "\n".join(table_strings)


# Test sql_to_table function with sql_context has some NULL columns
sql_context = """
CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT);
INSERT INTO salesperson (region, name) VALUES (1, 'John Doe'), (2, 'Jane Smith');
CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE);
INSERT INTO timber_sales (sales_id, salesperson_id, sale_date) VALUES (1, 1, 120), (2, 1, 150), (3, 2, 180);
"""

table_string = sql_to_table(sql_context)
print(table_string)

# Test the function on a sql_context sample
data_tables = sql_to_table(merged_ds[0]["sql_context"])
print(data_tables)


# Defines input and output of the instruction fine-tuning dataset
def prepare_instruct(dataset):
    # Giả sử bạn đã tải dataset ban đầu
    original_dataset = dataset

    # Chuyển đổi dataset thành DataFrame để xử lý
    df = original_dataset.to_pandas()

    # Khởi tạo tqdm cho pandas
    tqdm.pandas()

    # Tạo cột input bằng cách nối sql_prompt với sql_context
    df["prompt"] = df.progress_apply(
        lambda row: "Database's schema:\n\n"
        + sql_to_table(row["sql_context"]).strip()
        + "\n\n"
        + "Question:\n"
        + row["sql_prompt"].strip(),
        axis=1,
    )

    # df = df[["input", "sql"]]  # Giữ lại các cột cần thiết

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
# 5. Instruction fine-tune
# =============================================================================

# 5.1. Cấu hình tham số fine-tuning
# Login to HuggingFace (using token from environment)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables. Please check your .env file.")
login(hf_token)

hf_username = os.getenv("HUGGINGFACE_USERNAME")
if not hf_username:
    raise ValueError("HUGGINGFACE_USERNAME not found in environment variables. Please check your .env file.")

instruct_finetune_args = UnslothTrainingArguments(
    output_dir="/kaggle/working/finetune",
    seed=3407,
    logging_steps=200,
    save_steps=200,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=f"{hf_username}/XiYanSQL-QwenCoder-3B-2502-100kSQL_finetuned",  # Using username from environment
    hub_private_repo=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Fixed major bug in latest Unsloth
    warmup_steps=5,
    num_train_epochs=1,  # Set this for 1 full training run.
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    learning_rate=2e-5,
    embedding_learning_rate=2e-6,
    optim="adamw_8bit",
    lr_scheduler_type="linear",
    weight_decay=0.01,
    report_to="none",  # Use this for WandB etc
)

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

# 5.2. Tiến hành fine-tune
if __name__ == "__main__":
    print("Starting fine-tuning...")
    instruct_finetune_trainer_stats = instruct_finetune_trainer.train()
    print("Fine-tuning completed!")
