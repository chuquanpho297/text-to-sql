# SQL Generator Chat UI (Gradio)

A modern web interface for generating SQL queries from natural language using a fine-tuned XiYanSQL model.

[View on GitHub](https://github.com/quanminh6g123/text-to-sql)

## Features

- 🎯 **Natural Language to SQL**: Convert questions to SQL queries
- 🗄️ **In-Memory Database**: Automatic SQLite database creation from schema
- ▶️ **Query Execution**: Run generated SQL and view formatted results  
- ⚡ **Fast & Lightweight**: Built with Gradio for optimal ML model performance  
- 💡 **Example Schemas**: Pre-loaded database examples
- 🔒 **Secure**: Environment-based token management + safe in-memory testing
- 📱 **Responsive**: Works on desktop and mobile
- 🌐 **Easy Sharing**: Built-in public URL sharing with Gradio
- 🐳 **Docker Ready**: Easy deployment with Docker

## Quick Start

### Method 1: Direct Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Hugging Face token and username

# Run the app
python app_gradio.py
```

### Method 2: Docker

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:7860
```

## Environment Variables

Create a `.env` file with:

```env
HUGGINGFACE_TOKEN=your_token_here
HUGGINGFACE_USERNAME=your_username_here
```

## Model Information

- **Model**: `hng229/XiYanSQL-QwenCoder-3B-2502-100kSQL_finetuned`
- **Base**: XiYanSQL-QwenCoder-3B-2502
- **Fine-tuned on**: 100k SQL text-to-SQL dataset
- **Purpose**: Converting natural language questions to SQL queries

## Usage

1. **Load the Model**: Click "Load Model" to initialize your fine-tuned model
2. **Provide Database Schema**: Enter your database schema in CSV format or use an example
3. **Ask a Question**: Type your natural language question about the data
4. **Generate SQL**: Click "Generate SQL" to get the SQL query
5. **Review and Use**: Copy the generated SQL and use it in your database

## 📊 Example Schemas

The interface includes several pre-loaded examples:

- **E-commerce Database**: Customers, products, and orders
- **Library Management**: Books, members, and loans
- **Sales Database**: Salespeople and sales records

## 🔧 Configuration

### Environment Variables

- `HUGGINGFACE_TOKEN`: Your Hugging Face API token (required if the model is private or for uploading models/datasets).
- `HUGGINGFACE_USERNAME`: Your Hugging Face username (required for uploading models/datasets).
- `WANDB_PROJECT` (Optional): Your Weights & Biases project name for logging fine-tuning experiments (e.g., `xiyanSQL-32b-finetuning`). Used by `adjusted-32b-finetuning.py`.
- `WANDB_ENTITY` (Optional): Your Weights & Biases entity (username or team name). Used by `adjusted-32b-finetuning.py`.

### Model Settings

The model is configured with these default settings:

- `max_new_tokens`: 256
- `temperature`: 0.1
- `top_p`: 0.95

You can modify these in `app_gradio.py` for different generation behavior.

## 🌐 Deployment Options

### Local Development

```bash
python app_gradio.py
```

### Public Sharing

The Gradio app includes `share=True` which creates a public URL for easy sharing.

### Production Deployment

For production deployment, consider:

- Using a proper WSGI server (e.g., Gunicorn)
- Setting up reverse proxy (e.g., Nginx)
- Configuring proper environment variables
- Using containerization (Docker)

## 🎨 Interface Features

### Gradio Interface

- 🎨 Modern, responsive design
- 📋 Integrated example schemas
- ⚡ Quick model loading
- 🌐 Share functionality
- 💡 Usage tips and examples
- 🔧 Clear and regenerate options

## 🔍 Troubleshooting

### Model Loading Issues

- Ensure you have sufficient RAM/VRAM
- Check your Hugging Face token if the model is private
- Try using CPU inference if GPU memory is insufficient

### Performance Optimization

- Use GPU for faster inference
- Adjust batch size based on available memory
- Consider using quantized models for lower memory usage

### Common Errors

- **Token Error**: Make sure your `.env` file contains valid `HUGGINGFACE_TOKEN`
- **Memory Error**: Try closing other applications or use CPU inference
- **Import Error**: Ensure all requirements are installed with `pip install -r requirements.txt`

## 📁 Project Structure

### Core Application Files

- **`app_gradio.py`** - Main Gradio web interface for the SQL generator
  - Provides a modern web UI for natural language to SQL conversion
  - Handles model loading, tokenization, and inference
  - Includes pre-loaded database schema examples
  - Features responsive design with sharing capabilities

- **`mschema_implementation.py`** - M-Schema format converter
  - Converts SQL DDL statements to M-Schema format for enhanced training
  - Parses CREATE TABLE and INSERT statements
  - Formats database schemas according to XiYan-SQL M-Schema specification
  - Used by both the fine-tuning script and the main application

### Training and Fine-tuning

- **`adjusted-32b-finetuning.py`** - Fine-tuning script for XiYanSQL models
  - Specifically targets the `XGenerationLab/XiYanSQL-QwenCoder-32B-2412` model.
  - Implements 4-bit quantization and an adjusted sequence length (e.g., 1024) for memory efficiency.
  - Optimized for single GPU training (H100/A100 80GB recommended, requires 64GB+ VRAM).
  - Features dynamic batch size adjustment based on detected GPU type.
  - Integrated with Weights & Biases for experiment tracking (relies on `WANDB_PROJECT` and optional `WANDB_ENTITY` environment variables).

### Configuration Files

- **`requirements.txt`** - Python dependencies
  - Lists all required packages for the project
  - Includes ML libraries (transformers, torch, gradio)
  - Includes utility packages (pandas, sqlparse, python-dotenv)
  - Training dependencies (unsloth, wandb, huggingface_hub, bitsandbytes)

- **`.env.example`** - Environment variables template
  - Template for required environment configuration.
  - Includes placeholders for `HUGGINGFACE_TOKEN`, `HUGGINGFACE_USERNAME`.
  - Includes commented-out placeholders for optional Weights & Biases configuration (`WANDB_PROJECT`, `WANDB_ENTITY`) used during model fine-tuning.

### Additional Files

- **`README.md`** - This documentation file

## 🚀 Getting Started

### For Using the Web Interface

1. Install dependencies: `pip install -r requirements.txt`
2. Copy and configure environment: `cp .env.example .env`
3. Run the application: `python app_gradio.py`

### For Fine-tuning Models

1. Set up environment variables in `.env`
2. Ensure you have adequate GPU resources (H100/A100 recommended)
3. Run the fine-tuning script: `python adjusted-32b-finetuning.py`

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the interface.

## 📄 License

This project is open source. Please check the model license on Hugging Face for usage restrictions.

## 🆕 New Features (Database Integration)

- **In-Memory SQLite Database**: Automatically creates a SQLite database in memory from your schema
- **Query Execution**: Execute generated SQL queries and see real results
- **Data Validation**: Test your queries against actual data before using in production
- **Safe Testing Environment**: All operations run in memory - no persistent changes

### How It Works

1. **Load Schema**: When you provide a database schema, the app creates an in-memory SQLite database
2. **Initialize Data**: All CREATE TABLE and INSERT statements are executed to populate the database
3. **Generate SQL**: The AI generates SQL queries based on your questions
4. **Execute & View**: Click "Execute SQL" to run the query and see formatted results
