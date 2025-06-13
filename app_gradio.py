#!/usr/bin/env python3
"""
SQL Generation Chat UI - Gradio Version
Lightweight web interface for the fine-tuned XiYanSQL model
"""

import gradio as gr
import os
import re
import sqlite3
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import sqlparse  # For better SQL formatting and validation
from mschema_implementation import sql_to_mschema

# Load environment variables
load_dotenv()

# Global variables for model and LangChain pipeline
model = None
tokenizer = None
llm_pipeline = None
llm_chain = None

# Global variable for in-memory SQLite database
db_connection = None
current_schema_sql = None


def load_model():
    """Load the fine-tuned model and tokenizer with LangChain pipeline"""
    global model, tokenizer, llm_pipeline, llm_chain
    try:
        model_name = "hng229/XiYanSQL-QwenCoder-3B-2502-100kSQL_finetuned"

        # Check if we have a HuggingFace token
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        print("HuggingFace Token:", hf_token)
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
            template="""You are now a SQL data analyst, and you are given a database schema as follows:

【Schema】
{schema}

【Question】
{question}

Please read and understand the database schema carefully, and generate an executable SQL based on the user's question. The generated SQL is protected by ```sql and ```.
""",
        )

        # Create LCEL chain (modern LangChain syntax)
        llm_chain = prompt_template | llm_pipeline

        device_info = f"({model.device})" if hasattr(model, "device") else ""
        yield f"✅ Model and LangChain pipeline loaded successfully! {device_info}"

    except Exception as e:
        yield f"❌ Error loading model: {str(e)}"


def csv_to_sql_context(csv_schema):
    """Convert CSV format schema to SQL context for M-Schema processing"""
    if not csv_schema.strip():
        return ""

    sql_parts = []
    current_table = None

    lines = csv_schema.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if this is a table declaration
        if line.startswith("Table:"):
            current_table = line.replace("Table:", "").strip()
            continue

        # If we have a current table, process the data
        if current_table:
            # First non-table line should be column headers
            if "," in line and not any(char.isdigit() for char in line.split(",")[0]):
                # This is likely a header row
                columns = [col.strip() for col in line.split(",")]

                # Generate CREATE TABLE statement
                create_cols = []
                for col in columns:
                    # Simple type inference
                    if "id" in col.lower():
                        create_cols.append(f"{col} INTEGER")
                    elif "date" in col.lower():
                        create_cols.append(f"{col} DATE")
                    elif (
                        "price" in col.lower()
                        or "amount" in col.lower()
                        or "salary" in col.lower()
                    ):
                        create_cols.append(f"{col} REAL")
                    else:
                        create_cols.append(f"{col} TEXT")

                # Add PRIMARY KEY to first column if it contains 'id'
                if "id" in columns[0].lower():
                    create_cols[0] += " PRIMARY KEY"

                sql_parts.append(
                    f"CREATE TABLE {current_table} ({', '.join(create_cols)});"
                )

            elif "," in line:
                # This is likely a data row
                values = [val.strip() for val in line.split(",")]

                # Quote non-numeric values
                quoted_values = []
                for val in values:
                    if val.lower() == "null":
                        quoted_values.append("NULL")
                    elif val.isdigit() or (
                        val.replace(".", "").isdigit() and val.count(".") <= 1
                    ):
                        quoted_values.append(val)
                    else:
                        quoted_values.append(f"'{val}'")

                # Get column names from the CREATE TABLE statement
                for sql_part in reversed(sql_parts):
                    if f"CREATE TABLE {current_table}" in sql_part:
                        # Extract column names from CREATE TABLE
                        create_part = sql_part.split("(")[1].split(")")[0]
                        columns = [col.split()[0] for col in create_part.split(",")]

                        sql_parts.append(
                            f"INSERT INTO {current_table} ({', '.join(columns)}) VALUES ({', '.join(quoted_values)});"
                        )
                        break

    return "\n".join(sql_parts)


def parse_sql_schema(schema_text, db_name_input=None):
    """Parse SQL schema text to extract database name and SQL context"""
    schema_text = schema_text.strip()

    # Use provided db_name_input if available, otherwise extract from schema
    if db_name_input and db_name_input.strip():
        db_name = db_name_input.strip()
        sql_context = schema_text
    else:
        # Extract database name if present in schema text
        db_name = "database"  # default
        sql_context = schema_text

        if schema_text.startswith("DB_NAME:"):
            lines = schema_text.split("\n")
            first_line = lines[0].strip()
            if first_line.startswith("DB_NAME:"):
                db_name = first_line.replace("DB_NAME:", "").strip()
                # Remove the DB_NAME line and rejoin the rest
                sql_context = "\n".join(lines[1:]).strip()

    return db_name, sql_context


def init_database_only(schema_text):
    """Initialize database from schema without generating SQL"""
    if not schema_text.strip():
        return "❌ Please provide a database schema."

    try:
        # Initialize the database with the schema
        db_success, db_message = init_database_from_schema(schema_text.strip())
        return db_message
    except Exception as e:
        return f"❌ Error initializing database: {str(e)}"


def generate_sql(schema_text, question, progress=gr.Progress()):
    """Generate SQL query using LangChain with the fine-tuned model and initialize database"""
    global llm_chain

    if llm_chain is None:
        return (
            "❌ Please load the model first by clicking 'Load Model' button.",
            "❌ No database loaded",
        )

    if not schema_text.strip():
        return "❌ Please provide a database schema.", "❌ No database loaded"

    if not question.strip():
        return "❌ Please provide a question.", "❌ No database loaded"

    try:
        progress(0.1, desc="Initializing database...")

        # Initialize the database with the schema
        db_success, db_message = init_database_from_schema(schema_text.strip())

        if not db_success:
            return db_message, "❌ Database initialization failed"

        progress(0.3, desc="Converting schema to M-Schema format...")

        # Check if input is in new SQL format or old CSV format
        if schema_text.strip().startswith("DB_NAME:") or "CREATE TABLE" in schema_text:
            # New SQL format
            db_name, sql_context = parse_sql_schema(schema_text.strip(), "")
        else:
            # Old CSV format - convert to SQL first
            sql_context = csv_to_sql_context(schema_text.strip())
            db_name = "database"

            if not sql_context:
                return (
                    "❌ Failed to parse the schema. Please check the format.",
                    "❌ Schema parsing failed",
                )

        # Convert to M-Schema format
        mschema_format = sql_to_mschema(sql_context, db_name)

        # Use LangChain to generate SQL
        progress(0.5, desc="Generating SQL with LangChain...")

        # Use the chain with the M-Schema formatted input
        result = llm_chain.invoke(
            {"schema": mschema_format, "question": question.strip()}
        )

        progress(0.8, desc="Processing output...")

        # Clean up the response
        sql_response = result.strip()

        print("Raw response: ", sql_response)

        # Remove any extra text that might be generated after the SQL
        # Split by common separators and take only the first part (the SQL)
        separators = [
            "Human:",
            "Assistant:",
            "Note:",
            "Explanation:",
            "\n\n",
            "I'm sorry",
            "I can't",
            "If you have",
        ]

        for separator in separators:
            if separator in sql_response:
                sql_response = sql_response.split(separator)[0].strip()
                break

        # Remove common prefixes
        prefixes_to_remove = ["SQL Query:", "SQL:", "Query:", "Answer:"]
        for prefix in prefixes_to_remove:
            if sql_response.startswith(prefix):
                sql_response = sql_response[len(prefix) :].strip()

        # Remove any trailing text after semicolon if it's not SQL
        if ";" in sql_response:
            parts = sql_response.split(";")
            if len(parts) > 1:
                # Keep the SQL part (first part + semicolon)
                sql_response = parts[0].strip() + ";"

        print("Cleaned response: ", sql_response)

        # If response is empty or too short, return an error
        if not sql_response or len(sql_response) < 5:
            return (
                "❌ Failed to generate SQL. Please try rephrasing your question.",
                db_message,
            )

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

        return "\n".join(result_parts), db_message

    except Exception as e:
        return f"❌ Error generating SQL: {str(e)}", "❌ Error occurred"


def execute_generated_sql(sql_query):
    """Execute the generated SQL query and return formatted results"""
    if not sql_query.strip():
        return "❌ No SQL query to execute.", ""

    # Execute the query
    success, message, df = execute_sql_query(sql_query)

    if not success:
        return message, ""

    if df is None:
        return message, ""

    # Format results for display
    formatted_results = format_query_results(df)

    return message, formatted_results


# Example schemas
EXAMPLE_SCHEMAS = {
    "E-commerce Database": {
        "db_name": "ecommerce",
        "schema": """CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        phone TEXT
    );

    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT,
        price REAL,
        category TEXT
    );

    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        order_date DATE,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );

    INSERT INTO customers (customer_id, name, email, phone) VALUES 
    (1, 'John Doe', 'john@email.com', '123-456-7890'),
    (2, 'Jane Smith', 'jane@email.com', '098-765-4321');

    INSERT INTO products (product_id, name, price, category) VALUES 
    (1, 'Laptop', 999.99, 'Electronics'),
    (2, 'Book', 29.99, 'Education'),
    (3, 'Phone', 699.99, 'Electronics');

    INSERT INTO orders (order_id, customer_id, product_id, quantity, order_date) VALUES 
    (1, 1, 1, 1, '2024-01-15'),
    (2, 2, 2, 2, '2024-01-16'),
    (3, 1, 3, 1, '2024-01-17');""",
    },
    "Library Management": {
        "db_name": "library",
        "schema": """CREATE TABLE books (
        book_id INTEGER PRIMARY KEY,
        title TEXT,
        author TEXT,
        isbn TEXT,
        available INTEGER
    );

    CREATE TABLE members (
        member_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        join_date DATE
    );

    CREATE TABLE loans (
        loan_id INTEGER PRIMARY KEY,
        book_id INTEGER,
        member_id INTEGER,
        loan_date DATE,
        return_date DATE,
        FOREIGN KEY (book_id) REFERENCES books(book_id),
        FOREIGN KEY (member_id) REFERENCES members(member_id)
    );

    INSERT INTO books (book_id, title, author, isbn, available) VALUES 
    (1, 'Python Programming', 'John Author', '978-1234567890', 1),
    (2, 'Data Science', 'Jane Writer', '978-0987654321', 0),
    (3, 'Web Development', 'Bob Coder', '978-1122334455', 1);

    INSERT INTO members (member_id, name, email, join_date) VALUES 
    (1, 'Alice Johnson', 'alice@email.com', '2023-01-01'),
    (2, 'Bob Wilson', 'bob@email.com', '2023-02-15');

    INSERT INTO loans (loan_id, book_id, member_id, loan_date, return_date) VALUES 
    (1, 2, 1, '2024-01-10', NULL),
    (2, 1, 2, '2024-01-05', '2024-01-15');""",
    },
    "Sales Database": {
        "db_name": "sales",
        "schema": """CREATE TABLE salespeople (
        salesperson_id INTEGER PRIMARY KEY,
        name TEXT,
        region TEXT
    );

    CREATE TABLE sales (
        sale_id INTEGER PRIMARY KEY,
        salesperson_id INTEGER,
        amount REAL,
        sale_date DATE,
        FOREIGN KEY (salesperson_id) REFERENCES salespeople(salesperson_id)
    );

    INSERT INTO salespeople (salesperson_id, name, region) VALUES 
    (1, 'Tom Wilson', 'North'),
    (2, 'Lisa Brown', 'South'),
    (3, 'Mike Johnson', 'East');

    INSERT INTO sales (sale_id, salesperson_id, amount, sale_date) VALUES 
    (1, 1, 15000, '2024-01-15'),
    (2, 2, 22000, '2024-01-16'),
    (3, 1, 18000, '2024-01-17');""",
    },
    "HR Management": {
        "db_name": "hr_system",
        "schema": """CREATE TABLE employees (
        employee_id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary REAL,
        hire_date DATE
    );

    CREATE TABLE departments (
        department_id INTEGER PRIMARY KEY,
        department_name TEXT,
        manager_id INTEGER,
        FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
    );

    CREATE TABLE projects (
        project_id INTEGER PRIMARY KEY,
        project_name TEXT,
        department_id INTEGER,
        budget REAL,
        start_date DATE,
        FOREIGN KEY (department_id) REFERENCES departments(department_id)
    );

    INSERT INTO employees (employee_id, name, department, salary, hire_date) VALUES 
    (1, 'Alice Smith', 'Engineering', 75000, '2023-01-15'),
    (2, 'Bob Johnson', 'Sales', 65000, '2023-02-01'),
    (3, 'Carol Davis', 'Marketing', 70000, '2023-03-10');

    INSERT INTO departments (department_id, department_name, manager_id) VALUES 
    (1, 'Engineering', 1),
    (2, 'Sales', 2),
    (3, 'Marketing', 3);

    INSERT INTO projects (project_id, project_name, department_id, budget, start_date) VALUES 
    (1, 'AI Platform', 1, 100000, '2024-01-01'),
    (2, 'Sales Campaign', 2, 50000, '2024-02-01');""",
    },
    "School Database": {
        "db_name": "school",
        "schema": """CREATE TABLE students (
        student_id INTEGER PRIMARY KEY,
        name TEXT,
        grade INTEGER,
        age INTEGER
    );

    CREATE TABLE courses (
        course_id INTEGER PRIMARY KEY,
        course_name TEXT,
        teacher TEXT,
        credits INTEGER
    );

    CREATE TABLE enrollments (
        enrollment_id INTEGER PRIMARY KEY,
        student_id INTEGER,
        course_id INTEGER,
        grade TEXT,
        semester TEXT,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );

    INSERT INTO students (student_id, name, grade, age) VALUES 
    (1, 'Emma Wilson', 10, 16),
    (2, 'Liam Brown', 11, 17),
    (3, 'Olivia Davis', 9, 15);

    INSERT INTO courses (course_id, course_name, teacher, credits) VALUES 
    (1, 'Mathematics', 'Mr. Smith', 3),
    (2, 'English', 'Ms. Johnson', 3),
    (3, 'Science', 'Dr. Brown', 4);

    INSERT INTO enrollments (enrollment_id, student_id, course_id, grade, semester) VALUES 
    (1, 1, 1, 'A', 'Fall2024'),
    (2, 1, 2, 'B+', 'Fall2024'),
    (3, 2, 1, 'A-', 'Fall2024');""",
    },
}


def load_example_schema_and_init_db(example_name):
    """Load an example schema and initialize the database"""
    try:
        # Validate example name
        if not example_name:
            return "", "❌ Please select an example from the dropdown"

        if example_name not in EXAMPLE_SCHEMAS:
            return "", f"❌ Example '{example_name}' not found"

        schema_data = EXAMPLE_SCHEMAS[example_name]
        if not schema_data:
            return "", f"❌ Example '{example_name}' is empty"

        schema_text = schema_data.get("schema", "")
        if not schema_text:
            return "", f"❌ Example '{example_name}' has no schema"

        print(f"Loading example: {example_name}")
        print(f"Schema length: {len(schema_text)}")

        # Initialize the database
        db_success, db_message = init_database_from_schema(schema_text)
        print(f"DB init result: {db_success}, message: {db_message}")

        if not db_success:
            return schema_text, f"❌ Database initialization failed: {db_message}"

        return schema_text, db_message

    except Exception as e:
        error_msg = f"❌ Error loading example '{example_name}': {str(e)}"
        print(error_msg)
        return "", error_msg


def load_example_schema(example_name):
    """Load an example schema"""
    example_data = EXAMPLE_SCHEMAS.get(example_name, {})
    if example_data:
        return example_data.get("db_name", ""), example_data.get("schema", "")
    return "", ""


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


def init_database_from_schema(schema_text):
    """Initialize in-memory SQLite database from schema text"""
    global db_connection, current_schema_sql

    try:
        # Close existing connection if any
        if db_connection:
            db_connection.close()

        # Create new in-memory database
        db_connection = sqlite3.connect(":memory:")
        cursor = db_connection.cursor()

        # Parse schema text
        if schema_text.strip().startswith("DB_NAME:") or "CREATE TABLE" in schema_text:
            # SQL format
            db_name, sql_context = parse_sql_schema(schema_text.strip())
            current_schema_sql = sql_context
        else:
            # CSV format - convert to SQL first
            sql_context = csv_to_sql_context(schema_text.strip())
            current_schema_sql = sql_context

            if not sql_context:
                return False, "❌ Failed to parse the schema. Please check the format."

        # Execute SQL statements
        sql_statements = current_schema_sql.split(";")

        for statement in sql_statements:
            statement = statement.strip()
            if statement:
                try:
                    cursor.execute(statement)
                except sqlite3.Error as e:
                    print(f"Warning: Error executing statement '{statement}': {e}")
                    continue

        db_connection.commit()

        # Get table names and row counts to verify
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]

        if not table_names:
            return False, "❌ No tables were created. Please check your schema."

        # Get row counts for each table
        table_info = []
        for table_name in table_names:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                table_info.append(f"{table_name} ({row_count} rows)")
            except sqlite3.Error:
                table_info.append(f"{table_name} (unknown rows)")

        success_message = "✅ Database initialized successfully!\n"
        success_message += f"📊 Tables created: {', '.join(table_info)}\n"
        success_message += "🚀 Ready to generate and execute SQL queries!"

        return True, success_message

    except Exception as e:
        return False, f"❌ Error initializing database: {str(e)}"


def execute_sql_query(sql_query):
    """Execute SQL query on the in-memory database and return results"""
    global db_connection

    if not db_connection:
        return False, "❌ No database connection. Please load a schema first.", None

    try:
        # Clean the SQL query
        cleaned_query = sql_query.strip()
        if cleaned_query.startswith("--"):
            # Remove comment lines
            lines = cleaned_query.split("\n")
            sql_lines = [line for line in lines if not line.strip().startswith("--")]
            cleaned_query = "\n".join(sql_lines).strip()

        # Remove trailing semicolon if present
        if cleaned_query.endswith(";"):
            cleaned_query = cleaned_query[:-1]

        if not cleaned_query:
            return False, "❌ Empty SQL query.", None

        cursor = db_connection.cursor()
        cursor.execute(cleaned_query)

        # Get column names
        column_names = (
            [description[0] for description in cursor.description]
            if cursor.description
            else []
        )

        # Fetch results
        results = cursor.fetchall()

        if not results:
            return True, "✅ Query executed successfully but returned no results.", None

        # Convert to pandas DataFrame for better display
        df = pd.DataFrame(results, columns=column_names)

        return True, f"✅ Query executed successfully. Found {len(results)} rows.", df

    except sqlite3.Error as e:
        return False, f"❌ SQL Error: {str(e)}", None
    except Exception as e:
        return False, f"❌ Error executing query: {str(e)}", None


def format_query_results(df):
    """Format query results for display"""
    if df is None or df.empty:
        return "No results to display."

    try:
        # Use tabulate for nice formatting
        formatted_table = tabulate(df, headers="keys", tablefmt="grid", showindex=False)

        # Add some summary information
        summary = f"Query Results ({len(df)} rows, {len(df.columns)} columns)\n"
        summary += "=" * 50 + "\n"
        summary += formatted_table

        return summary
    except Exception:
        # Fallback to simple string representation
        return f"Results ({len(df)} rows):\n\n{df.to_string(index=False)}"


# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="SQL Generator Chat",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        }
        .main-section {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
            border-radius: 10px !important;
            padding: 20px !important;
            margin: 10px 0 !important;
            border: 1px solid #3498db !important;
        }
        .main-section h3 {
            color: #3498db !important;
            margin-bottom: 15px !important;
            font-size: 18px !important;
        }
        .compact-row {
            display: flex;
            gap: 10px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .status-info {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%) !important;
            border-radius: 8px !important;
            padding: 12px !important;
            margin: 8px 0 !important;
            color: white !important;
            font-size: 14px !important;
        }
        .header-compact {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            padding: 20px !important;
            border-radius: 12px !important;
            text-align: center !important;
            margin-bottom: 15px !important;
        }
        .header-compact h1 {
            margin: 0 0 8px 0 !important;
            font-size: 2.2em !important;
            color: white !important;
        }
        .header-compact p {
            margin: 4px 0 !important;
            font-size: 14px !important;
        }
        .gr-textbox, .gr-code, .gr-dropdown {
            background: #34495e !important;
            border: 1px solid #3498db !important;
            border-radius: 6px !important;
            color: #ecf0f1 !important;
        }
        .gr-button {
            border-radius: 6px !important;
            font-weight: bold !important;
            padding: 8px 16px !important;
        }
        .sidebar-compact {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin: 10px 0 !important;
            border: 1px solid #3498db !important;
        }
        .example-compact {
            background: #2c3e50 !important;
            border: 1px solid #3498db !important;
            border-radius: 8px !important;
            padding: 12px !important;
            margin: 8px 0 !important;
            font-size: 13px !important;
            color: #ecf0f1 !important;
        }
        .example-compact h4 {
            color: #3498db !important;
            margin-bottom: 8px !important;
            font-size: 16px !important;
        }
        """,
    ) as demo:
        gr.HTML("""
        <div class="header-section">
            <h1>🗄️ SQL Generator Chat</h1>
            <p style="font-size: 18px;">Powered by XiYanSQL-QwenCoder-3B Fine-tuned Model with LangChain</p>
            <p style="font-size: 16px;">✨ In-Memory SQLite Database + Real Query Execution + Formatted Results ✨</p>
            <p style="font-size: 16px; font-weight: bold; color: #f39c12;">💡 Load Schema → Initialize Database → Generate SQL → Execute & See Results!</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Model loading section
                with gr.Group(elem_classes="main-section"):
                    gr.HTML(
                        "<h3 style='color: #2E86AB; margin-bottom: 20px;'>🚀 Model Setup</h3>"
                    )
                    with gr.Row(elem_classes="button-row"):
                        load_btn = gr.Button("Load Model", variant="primary", size="lg")
                        clear_memory_btn = gr.Button(
                            "Clear Memory", variant="secondary", size="sm"
                        )
                        info_btn = gr.Button(
                            "Model Info", variant="secondary", size="sm"
                        )
                    load_status = gr.Textbox(
                        label="Model Status",
                        interactive=False,
                        lines=3,
                        elem_classes="status-box",
                    )

                # Main interface
                with gr.Group(elem_classes="main-section"):
                    gr.HTML(
                        "<h3 style='color: #2E86AB; margin-bottom: 20px;'>💬 Generate SQL Query</h3>"
                    )

                    with gr.Row():
                        with gr.Column():
                            schema_input = gr.Textbox(
                                label="📊 Database Schema (SQL format)",
                                placeholder="Paste your database schema here (SQL CREATE TABLE and INSERT statements)...",
                                lines=12,
                                max_lines=20,
                                show_label=True,
                            )
                        with gr.Column():
                            question_input = gr.Textbox(
                                label="❓ Your Question",
                                placeholder="What data do you want to query? Example: 'Show me all customers' or 'What's the total revenue?'",
                                lines=6,
                                max_lines=12,
                                show_label=True,
                            )

                    with gr.Row(elem_classes="button-row"):
                        init_db_btn = gr.Button(
                            "🗄️ Initialize Database", variant="secondary", size="lg"
                        )
                        generate_btn = gr.Button(
                            "🚀 Generate SQL", variant="primary", size="lg"
                        )
                        execute_btn = gr.Button(
                            "▶️ Execute SQL", variant="success", size="lg"
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear All", variant="secondary", size="sm"
                        )

                    # SQL Output
                    sql_output = gr.Code(
                        label="📝 Generated SQL Query",
                        language="sql",
                        lines=8,
                        show_label=True,
                    )

                    # Database and execution status
                    with gr.Row():
                        with gr.Column():
                            db_status = gr.Textbox(
                                label="🗄️ Database Status",
                                interactive=False,
                                lines=3,
                                placeholder="Database not initialized...",
                                elem_classes="status-box",
                                show_label=True,
                            )
                        with gr.Column():
                            results_status = gr.Textbox(
                                label="⚡ Execution Status",
                                interactive=False,
                                lines=2,
                                elem_classes="status-box",
                                show_label=True,
                            )

                    # Query results
                    query_results = gr.Textbox(
                        label="📋 Query Results",
                        lines=12,
                        max_lines=20,
                        interactive=False,
                        elem_classes="status-box",
                        show_label=True,
                    )

            with gr.Column(scale=1):
                # Example schemas
                with gr.Group(elem_classes="sidebar-section"):
                    gr.HTML(
                        "<h3 style='color: #2E86AB; margin-bottom: 15px;'>📋 Example Schemas</h3>"
                    )
                    example_dropdown = gr.Dropdown(
                        choices=list(EXAMPLE_SCHEMAS.keys()),
                        label="Choose an example",
                        value=None,
                        interactive=True,
                        show_label=True,
                    )
                    load_example_btn = gr.Button(
                        "📥 Load Example", variant="secondary", size="lg"
                    )

                # Example questions
                with gr.Group(elem_classes="sidebar-section"):
                    gr.HTML("""
                    <div class="example-box" style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important; border: 2px solid #3498db !important; padding: 25px !important; margin: 15px 0 !important; border-radius: 15px !important;">
                        <h4 style="color: #3498db !important; font-weight: bold !important; font-size: 20px !important; margin-bottom: 20px !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;">💡 Example Questions</h4>
                        <ul style="color: #ecf0f1 !important; margin: 15px 0 !important; padding-left: 25px !important;">
                            <li style="color: #ecf0f1 !important; margin: 12px 0 !important; font-size: 14px !important;">"Show total sales by region"</li>
                            <li style="color: #ecf0f1 !important; margin: 12px 0 !important; font-size: 14px !important;">"Find customers with most orders"</li>
                            <li style="color: #ecf0f1 !important; margin: 12px 0 !important; font-size: 14px !important;">"List products in Electronics category"</li>
                            <li style="color: #ecf0f1 !important; margin: 12px 0 !important; font-size: 14px !important;">"Calculate average order value"</li>
                        </ul>
                    </div>
                    """)

        # Event handlers
        load_btn.click(load_model, outputs=load_status, show_progress=True)

        clear_memory_btn.click(clear_memory, outputs=load_status)

        info_btn.click(get_model_info, outputs=load_status)

        init_db_btn.click(
            init_database_only,
            inputs=[schema_input],
            outputs=[db_status],
            show_progress=True,
        )

        generate_btn.click(
            generate_sql,
            inputs=[schema_input, question_input],
            outputs=[sql_output, db_status],
            show_progress=True,
        )

        execute_btn.click(
            execute_generated_sql,
            inputs=[sql_output],
            outputs=[results_status, query_results],
            show_progress=True,
        )

        # Clear function
        def clear_all():
            global db_connection
            # Close database connection
            if db_connection:
                db_connection.close()
                db_connection = None
            return "", "", "", "Database connection closed", "", ""

        clear_btn.click(
            clear_all,
            outputs=[
                schema_input,
                question_input,
                sql_output,
                db_status,
                results_status,
                query_results,
            ],
        )

        load_example_btn.click(
            load_example_schema_and_init_db,
            inputs=example_dropdown,
            outputs=[schema_input, db_status],
        )

        # Example interactions
        gr.Examples(
            examples=[
                [
                    "ecommerce",
                    EXAMPLE_SCHEMAS["E-commerce Database"]["schema"],
                    "Show me the total revenue by product category",
                ],
                [
                    "library",
                    EXAMPLE_SCHEMAS["Library Management"]["schema"],
                    "Find all books that are currently borrowed",
                ],
                [
                    "sales",
                    EXAMPLE_SCHEMAS["Sales Database"]["schema"],
                    "Which salesperson has the highest total sales?",
                ],
                [
                    "hr_system",
                    EXAMPLE_SCHEMAS["HR Management"]["schema"],
                    "List all employees in Engineering department with their salaries",
                ],
                [
                    "school",
                    EXAMPLE_SCHEMAS["School Database"]["schema"],
                    "Show average grade for each course",
                ],
            ],
            inputs=[schema_input, question_input],
            outputs=[sql_output, db_status],
            fn=generate_sql,
            cache_examples=False,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
