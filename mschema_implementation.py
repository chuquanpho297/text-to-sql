#!/usr/bin/env python3
"""
M-Schema Implementation for SQL Finetuning

This module provides functions to convert SQL context from DDL format
to M-Schema format for enhanced text-to-SQL finetuning performance.

Based on XiYan-SQL M-Schema specification:
https://github.com/XGenerationLab/M-Schema
"""

import re
from typing import Dict, List


def sql_to_mschema(context: str, db_name: str = "database") -> str:
    """
    Convert SQL context to M-Schema format.

    Args:
        context: SQL script containing CREATE TABLE and INSERT statements
        db_name: Name of the database (default: "database")

    Returns:
        String representation in M-Schema format
    """
    sql_script = context

    # Extract CREATE TABLE and INSERT statements
    create_pattern = re.compile(r"CREATE TABLE (\w+) \((.*?)\);", re.DOTALL)
    insert_pattern = re.compile(r"INSERT INTO (\w+) \((.*?)\) VALUES (.*?);", re.DOTALL)

    # Parse table schemas and data
    table_schemas = {}
    table_data = {}

    # Process CREATE TABLE statements
    for match in create_pattern.finditer(sql_script):
        table_name = match.group(1)
        columns_def = match.group(2)

        # Parse column definitions
        columns = []
        for col_def in columns_def.split(","):
            col_def = col_def.strip()
            col_parts = col_def.split()

            if len(col_parts) >= 2:
                col_name = col_parts[0]
                col_type = col_parts[1]

                # Check for constraints
                is_primary = "PRIMARY KEY" if "PRIMARY KEY" in col_def.upper() else ""
                is_foreign = "FOREIGN KEY" in col_def.upper()

                columns.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "primary": is_primary,
                        "foreign": is_foreign,
                        "definition": col_def,
                    }
                )

        table_schemas[table_name] = columns
        table_data[table_name] = []

    # Process INSERT statements to get sample data
    for match in insert_pattern.finditer(sql_script):
        table_name = match.group(1)
        if table_name not in table_schemas:
            continue

        columns_list = [col.strip() for col in match.group(2).split(",")]
        values = match.group(3)

        # Extract value rows
        rows = re.findall(r"\((.*?)\)", values)
        for row in rows:
            row_values = [val.strip().strip("'\"") for val in row.split(",")]
            row_dict = {}
            for col, val in zip(columns_list, row_values):
                row_dict[col] = val
            table_data[table_name].append(row_dict)

    # Build M-Schema representation
    mschema_parts = []

    # Database header
    if db_name:
        mschema_parts.append(f"[DATABASE] {db_name}")
    mschema_parts.append("[SCHEMA]")
    mschema_parts.append("")

    # Process each table
    for table_name, columns in table_schemas.items():
        # Table header
        mschema_parts.append(f"# Table {table_name}")

        # Column definitions in M-Schema format
        column_tuples = []
        for col in columns:
            col_name = col["name"]
            col_type = col["type"]
            primary_flag = col["primary"]

            # Get sample values for this column
            sample_values = get_sample_values(table_data.get(table_name, []), col_name)

            # Format as M-Schema tuple
            column_tuple = f"({col_name}, {col_type}, {primary_flag}, {sample_values})"
            column_tuples.append(column_tuple)

        # Add column list
        if column_tuples:
            mschema_parts.append("[" + ",\n ".join(column_tuples) + "]")
        mschema_parts.append("")

    # Foreign keys section
    foreign_keys = detect_foreign_keys(table_schemas, sql_script)
    mschema_parts.append("【Foreign Keys】")
    for fk in foreign_keys:
        mschema_parts.append(fk)

    return "\n".join(mschema_parts)


def get_sample_values(
    table_data: List[Dict], column_name: str, max_samples: int = 5
) -> List:
    """
    Extract sample values for a column from table data.

    Args:
        table_data: List of row dictionaries
        column_name: Name of the column
        max_samples: Maximum number of sample values to return

    Returns:
        List of sample values
    """
    if not table_data:
        return []

    values = []
    for row in table_data:
        if column_name in row and row[column_name] != "NULL":
            value = row[column_name]
            if value not in values:
                values.append(value)
                if len(values) >= max_samples:
                    break

    return values


def detect_foreign_keys(table_schemas: Dict, sql_script: str) -> List[str]:
    """
    Detect foreign key relationships from SQL script.

    Args:
        table_schemas: Dictionary of table schemas
        sql_script: Original SQL script

    Returns:
        List of foreign key relationship strings
    """
    foreign_keys = []

    # Look for explicit FOREIGN KEY constraints
    fk_pattern = re.compile(
        r"FOREIGN KEY \((\w+)\) REFERENCES (\w+)\((\w+)\)", re.IGNORECASE
    )

    current_table = None
    for line in sql_script.split("\n"):
        # Track current table
        create_match = re.search(r"CREATE TABLE (\w+)", line, re.IGNORECASE)
        if create_match:
            current_table = create_match.group(1)

        # Look for foreign key constraints
        fk_match = fk_pattern.search(line)
        if fk_match and current_table:
            fk_column = fk_match.group(1)
            ref_table = fk_match.group(2)
            ref_column = fk_match.group(3)

            foreign_keys.append(
                f"{current_table}.{fk_column} -> {ref_table}.{ref_column}"
            )

    # Heuristic detection based on column names
    # Look for columns ending with "_id" that might reference other tables
    for table_name, columns in table_schemas.items():
        for col in columns:
            col_name = col["name"]
            if col_name.endswith("_id") and col_name != "id":
                # Try to find referenced table
                ref_table_name = col_name[:-3]  # Remove "_id"
                if ref_table_name in table_schemas:
                    fk_relationship = f"{table_name}.{col_name} -> {ref_table_name}.id"
                    if fk_relationship not in foreign_keys:
                        foreign_keys.append(fk_relationship)

    return foreign_keys


def prepare_instruct_mschema(dataset):
    """
    Modified prepare_instruct function to use M-Schema format.

    Args:
        dataset: Hugging Face dataset with sql_context and sql_prompt columns

    Returns:
        Processed dataset with M-Schema formatted prompts
    """
    from tqdm.auto import tqdm
    from datasets import Dataset

    # Convert to pandas for processing
    df = dataset.to_pandas()
    tqdm.pandas()

    # Create prompts with M-Schema format
    df["prompt"] = df.progress_apply(
        lambda row: "Database's schema:\n\n"
        + sql_to_mschema(row["sql_context"], row.get("domain", "database")).strip()
        + "\n\n"
        + "Question:\n"
        + row["sql_prompt"].strip(),
        axis=1,
    )

    # Convert back to Hugging Face dataset
    processed_dataset = Dataset.from_pandas(df)
    return processed_dataset


# Example usage and testing
if __name__ == "__main__":
    # Example SQL context for testing
    test_sql = """
    CREATE TABLE students (
        student_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        enrollment_date DATE
    );

    CREATE TABLE courses (
        course_id INTEGER PRIMARY KEY,
        course_name TEXT,
        credits INTEGER,
        instructor TEXT
    );

    CREATE TABLE enrollments (
        enrollment_id INTEGER PRIMARY KEY,
        student_id INTEGER,
        course_id INTEGER,
        grade TEXT,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );

    INSERT INTO students (student_id, name, email, enrollment_date) VALUES 
    (1, 'John Doe', 'john@email.com', '2023-09-01'),
    (2, 'Jane Smith', 'jane@email.com', '2023-09-15'),
    (3, 'Bob Johnson', 'bob@email.com', '2023-08-20');

    INSERT INTO courses (course_id, course_name, credits, instructor) VALUES 
    (101, 'Math 101', 3, 'Dr. Smith'),
    (102, 'Physics 201', 4, 'Prof. Johnson'),
    (103, 'Chemistry 301', 3, 'Dr. Brown');

    INSERT INTO enrollments (enrollment_id, student_id, course_id, grade) VALUES 
    (1, 1, 101, 'A'),
    (2, 2, 101, 'B+'),
    (3, 1, 102, 'A-'),
    (4, 3, 103, 'B');
    """

    print(sql_to_mschema(test_sql, "university_db"))
