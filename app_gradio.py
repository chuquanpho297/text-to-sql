#!/usr/bin/env python3
"""
SQL Generation Chat UI - Gradio Version
Lightweight web interface for the fine-tuned XiYanSQL model
"""

import gradio as gr
import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import sqlparse  # For better SQL formatting and validation

# Load environment variables
load_dotenv()

# Global variables for model and LangChain pipeline
model = None
tokenizer = None
llm_pipeline = None
llm_chain = None


def load_model():
    """Load the fine-tuned model and tokenizer with LangChain pipeline"""
    global model, tokenizer, llm_pipeline, llm_chain
    try:
        model_name = "hng229/XiYanSQL-QwenCoder-3B-2502-100kSQL_finetuned"

        # Check if we have a HuggingFace token
        hf_token = os.getenv("HUGGINGFACE_TOKEN")

        yield "🔄 Loading tokenizer..."

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token if hf_token else None
        )

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        yield "🔄 Loading model... (this may take a few minutes)"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token if hf_token else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Move to GPU if available
        if torch.cuda.is_available() and model.device.type == "cpu":
            model = model.cuda()

        yield "🔄 Setting up LangChain pipeline..."

        # Create HuggingFace pipeline for text generation
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Create LangChain HuggingFace pipeline
        llm_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)

        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["schema", "question"],
            template="""You are an AI assistant specialized in converting natural language questions into accurate SQL queries. Based on the question's context and the provided database schema, generate an optimized, concise, and syntactically correct SQL query. Do not explain, only return the SQL.

Database's schema:

{schema}

Question:
{question}

SQL Query:""",
        )

        # Create LCEL chain (modern LangChain syntax)
        llm_chain = prompt_template | llm_pipeline

        device_info = f"({model.device})" if hasattr(model, "device") else ""
        yield f"✅ Model and LangChain pipeline loaded successfully! {device_info}"

    except Exception as e:
        yield f"❌ Error loading model: {str(e)}"


def generate_sql(schema_text, question, progress=gr.Progress()):
    """Generate SQL query using LangChain with the fine-tuned model"""
    global llm_chain

    if llm_chain is None:
        return "❌ Please load the model first by clicking 'Load Model' button."

    if not schema_text.strip():
        return "❌ Please provide a database schema."

    if not question.strip():
        return "❌ Please provide a question."

    try:
        progress(0.2, desc="Preparing inputs...")

        # Use LangChain to generate SQL
        progress(0.5, desc="Generating SQL with LangChain...")

        # Use the chain with the inputs (modern LCEL syntax)
        result = llm_chain.invoke(
            {"schema": schema_text.strip(), "question": question.strip()}
        )

        progress(0.8, desc="Processing output...")

        # Clean up the response
        sql_response = result.strip()

        print("response: ", sql_response)

        # Remove any extra text that might be generated
        if "SQL Query:" in sql_response:
            sql_response = sql_response.split("SQL Query:")[-1].strip()

        # If response is empty or too short, return an error
        if not sql_response or len(sql_response) < 5:
            return "❌ Failed to generate SQL. Please try rephrasing your question."

        # Format the SQL
        formatted_sql = format_sql(sql_response)

        # Validate the SQL
        is_valid, validation_msg = validate_sql(formatted_sql)

        # Create the final output with validation info
        result_parts = [f"-- Question: {question}"]

        if not is_valid:
            result_parts.append(f"-- ⚠️ Validation: {validation_msg}")
        elif "Warning" in validation_msg:
            result_parts.append(f"-- {validation_msg}")

        result_parts.append(formatted_sql)

        progress(1.0, desc="Complete!")

        return "\n".join(result_parts)

    except Exception as e:
        return f"❌ Error generating SQL: {str(e)}"


# Example schemas
EXAMPLE_SCHEMAS = {
    "E-commerce Database": """Table: customers
customer_id,name,email,phone
1,John Doe,john@email.com,123-456-7890
2,Jane Smith,jane@email.com,098-765-4321

Table: products
product_id,name,price,category
1,Laptop,999.99,Electronics
2,Book,29.99,Education
3,Phone,699.99,Electronics

Table: orders
order_id,customer_id,product_id,quantity,order_date
1,1,1,1,2024-01-15
2,2,2,2,2024-01-16
3,1,3,1,2024-01-17""",
    "Library Management": """Table: books
book_id,title,author,isbn,available
1,Python Programming,John Author,978-1234567890,1
2,Data Science,Jane Writer,978-0987654321,0
3,Web Development,Bob Coder,978-1122334455,1

Table: members
member_id,name,email,join_date
1,Alice Johnson,alice@email.com,2023-01-01
2,Bob Wilson,bob@email.com,2023-02-15

Table: loans
loan_id,book_id,member_id,loan_date,return_date
1,2,1,2024-01-10,NULL
2,1,2,2024-01-05,2024-01-15""",
    "Sales Database": """Table: salespeople
salesperson_id,name,region
1,Tom Wilson,North
2,Lisa Brown,South
3,Mike Johnson,East

Table: sales
sale_id,salesperson_id,amount,sale_date
1,1,15000,2024-01-15
2,2,22000,2024-01-16
3,1,18000,2024-01-17""",
    "HR Management": """Table: employees
employee_id,name,department,salary,hire_date
1,Alice Smith,Engineering,75000,2023-01-15
2,Bob Johnson,Sales,65000,2023-02-01
3,Carol Davis,Marketing,70000,2023-03-10

Table: departments
department_id,department_name,manager_id
1,Engineering,1
2,Sales,2
3,Marketing,3

Table: projects
project_id,project_name,department_id,budget,start_date
1,AI Platform,1,100000,2024-01-01
2,Sales Campaign,2,50000,2024-02-01""",
    "School Database": """Table: students
student_id,name,grade,age
1,Emma Wilson,10,16
2,Liam Brown,11,17
3,Olivia Davis,9,15

Table: courses
course_id,course_name,teacher,credits
1,Mathematics,Mr. Smith,3
2,English,Ms. Johnson,3
3,Science,Dr. Brown,4

Table: enrollments
enrollment_id,student_id,course_id,grade,semester
1,1,1,A,Fall2024
2,1,2,B+,Fall2024
3,2,1,A-,Fall2024""",
}


def load_example_schema(example_name):
    """Load an example schema"""
    return EXAMPLE_SCHEMAS.get(example_name, "")


def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "🧹 GPU memory cleared"
    return "💾 Running on CPU - no GPU memory to clear"


def get_model_info():
    """Get current model and LangChain pipeline information"""
    global model, tokenizer, llm_pipeline, llm_chain
    if model is None or llm_chain is None:
        return "❌ No model or LangChain pipeline loaded"

    device = getattr(model, "device", "unknown")
    model_size = (
        sum(p.numel() for p in model.parameters()) / 1e6
    )  # millions of parameters

    info = f"✅ Model loaded on {device}\n"
    info += f"📊 Parameters: ~{model_size:.1f}M\n"
    info += (
        f"🔗 LangChain Pipeline: {'✅ Active' if llm_pipeline else '❌ Not loaded'}\n"
    )
    info += f"⛓️ LangChain LCEL Chain: {'✅ Active' if llm_chain else '❌ Not loaded'}\n"

    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        info += f"🔧 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB"

    return info


def format_sql(sql_query):
    """Enhanced SQL formatting and validation"""
    if not sql_query:
        return sql_query

    try:
        # Use sqlparse for better formatting
        formatted = sqlparse.format(
            sql_query,
            reindent=True,
            keyword_case="upper",
            identifier_case="lower",
            strip_comments=False,
        )
        return formatted.strip()
    except Exception:
        # Fallback to basic formatting if sqlparse fails
        sql_query = " ".join(sql_query.split())

        # Basic SQL keyword formatting
        keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "INNER JOIN",
            "OUTER JOIN",
            "ON",
            "AND",
            "OR",
            "NOT",
            "IN",
            "LIKE",
            "BETWEEN",
            "IS",
            "NULL",
            "AS",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVG",
            "MAX",
            "MIN",
            "LIMIT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "ALTER",
            "DROP",
        ]

        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            sql_query = re.sub(pattern, keyword, sql_query, flags=re.IGNORECASE)

        return sql_query


def validate_sql(sql_query):
    """Basic SQL validation"""
    if not sql_query.strip():
        return False, "Empty query"

    try:
        # Parse the SQL to check for basic syntax errors
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            return False, "Unable to parse SQL"

        # Check for common dangerous operations in a basic way
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
        sql_upper = sql_query.upper()

        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return True, f"⚠️ Warning: Contains {keyword} operation"

        return True, "SQL appears valid"
    except Exception as e:
        return False, f"SQL validation error: {str(e)}"


# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="SQL Generator Chat",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .model-info {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .example-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """,
    ) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #2E86AB; margin-bottom: 10px;">🗄️ SQL Generator Chat</h1>
            <p style="color: #666; font-size: 18px;">Powered by XiYanSQL-QwenCoder-3B Fine-tuned Model with LangChain</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Model loading section
                with gr.Group():
                    gr.HTML("<h3>🚀 Model Setup</h3>")
                    with gr.Row():
                        load_btn = gr.Button("Load Model", variant="primary", size="lg")
                        clear_memory_btn = gr.Button(
                            "Clear Memory", variant="secondary", size="sm"
                        )
                        info_btn = gr.Button(
                            "Model Info", variant="secondary", size="sm"
                        )
                    load_status = gr.Textbox(label="Status", interactive=False, lines=3)

                # Main interface
                with gr.Group():
                    gr.HTML("<h3>💬 Generate SQL Query</h3>")

                    with gr.Row():
                        schema_input = gr.Textbox(
                            label="Database Schema (CSV format)",
                            placeholder="Paste your database schema here...",
                            lines=10,
                            max_lines=15,
                        )
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="What data do you want to query?",
                            lines=5,
                            max_lines=10,
                        )

                    generate_btn = gr.Button(
                        "🚀 Generate SQL", variant="primary", size="lg"
                    )
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary", size="sm")

                    sql_output = gr.Code(
                        label="Generated SQL Query", language="sql", lines=10
                    )

            with gr.Column(scale=1):
                # Example schemas
                with gr.Group():
                    gr.HTML("<h3>📋 Example Schemas</h3>")
                    example_dropdown = gr.Dropdown(
                        choices=list(EXAMPLE_SCHEMAS.keys()),
                        label="Choose an example",
                        value=None,
                    )
                    load_example_btn = gr.Button("Load Example", variant="secondary")

                # Model information
                gr.HTML("""
                <div class="model-info">
                    <h3>ℹ️ Model Information</h3>
                    <p><strong>Model:</strong> XiYanSQL-QwenCoder-3B-2502-100kSQL_finetuned</p>
                    <p><strong>Framework:</strong> LangChain + HuggingFace Transformers</p>
                    <p><strong>Purpose:</strong> Converting natural language to SQL queries</p>
                    <p><strong>Input:</strong> Database schema + Question</p>
                    <p><strong>Output:</strong> SQL query</p>
                </div>
                """)

                # Usage tips
                gr.HTML("""
                <div class="example-box">
                    <h4>📝 Usage Tips</h4>
                    <ul>
                        <li>Provide clear database schema in CSV format</li>
                        <li>Include table names, columns, and sample data</li>
                        <li>Ask specific questions about your data</li>
                        <li>Review generated SQL before executing</li>
                    </ul>
                </div>
                """)

                # Schema format guide
                gr.HTML("""
                <div class="example-box">
                    <h4>📊 Schema Format</h4>
                    <p><strong>Format:</strong> Table: table_name<br>
                    column1,column2,column3<br>
                    value1,value2,value3</p>
                    <p><strong>Example:</strong><br>
                    Table: users<br>
                    id,name,email<br>
                    1,John,john@email.com</p>
                </div>
                """)

                # Example questions
                gr.HTML("""
                <div class="example-box">
                    <h4>💡 Example Questions</h4>
                    <ul>
                        <li>"Show total sales by region"</li>
                        <li>"Find customers with most orders"</li>
                        <li>"List products in Electronics category"</li>
                        <li>"Calculate average order value"</li>
                    </ul>
                </div>
                """)

        # Event handlers
        load_btn.click(load_model, outputs=load_status, show_progress=True)

        clear_memory_btn.click(clear_memory, outputs=load_status)

        info_btn.click(get_model_info, outputs=load_status)

        generate_btn.click(
            generate_sql,
            inputs=[schema_input, question_input],
            outputs=sql_output,
            show_progress=True,
        )

        # Clear function
        def clear_all():
            return "", "", ""

        clear_btn.click(clear_all, outputs=[schema_input, question_input, sql_output])

        load_example_btn.click(
            load_example_schema, inputs=example_dropdown, outputs=schema_input
        )

        # Example interactions
        gr.Examples(
            examples=[
                [
                    EXAMPLE_SCHEMAS["E-commerce Database"],
                    "Show me the total revenue by product category",
                ],
                [
                    EXAMPLE_SCHEMAS["Library Management"],
                    "Find all books that are currently borrowed",
                ],
                [
                    EXAMPLE_SCHEMAS["Sales Database"],
                    "Which salesperson has the highest total sales?",
                ],
                [
                    EXAMPLE_SCHEMAS["HR Management"],
                    "List all employees in Engineering department with their salaries",
                ],
                [
                    EXAMPLE_SCHEMAS["School Database"],
                    "Show average grade for each course",
                ],
            ],
            inputs=[schema_input, question_input],
            outputs=sql_output,
            fn=generate_sql,
            cache_examples=False,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
