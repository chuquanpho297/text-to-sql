# SQL Generator Chat UI (Gradio)

A modern web interface for generating SQL queries from natural language using a fine-tuned XiYanSQL model.

## Features

- 🎯 **Natural Language to SQL**: Convert questions to SQL queries
- ⚡ **Fast & Lightweight**: Built with Gradio for optimal ML model performance  
- 💡 **Example Schemas**: Pre-loaded database examples
- 🔒 **Secure**: Environment-based token management
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
```
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
- `HUGGINGFACE_TOKEN`: Your Hugging Face API token (if model is private)
- `HUGGINGFACE_USERNAME`: Your Hugging Face username

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

## 📝 Model Information

- **Model**: hng229/XiYanSQL-QwenCoder-3B-2502-100kSQL_finetuned
- **Base Model**: XiYanSQL-QwenCoder-3B-2502
- **Fine-tuned on**: 100k SQL text-to-SQL dataset
- **Purpose**: Natural language to SQL query conversion

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the interface.

## 📄 License

This project is open source. Please check the model license on Hugging Face for usage restrictions.
