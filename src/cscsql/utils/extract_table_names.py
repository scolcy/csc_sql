import json
import re
import argparse
from typing import List, Dict, Any


def extract_table_names_from_sql(sql: str) -> List[str]:
    """
    Extract table names from SQL query.
    
    Args:
        sql (str): SQL query string
        
    Returns:
        List[str]: List of table names found in the SQL
    """
    # Remove extra whitespace and newlines
    sql = ' '.join(sql.split())
    
    # Pattern to match table names in FROM and JOIN clauses
    # This pattern looks for FROM or JOIN followed by optional schema name and table name
    pattern = r'(?:FROM|JOIN)\s+(?:`?(\w+)`?\.)?`?(\w+)`?'
    
    table_names = []
    matches = re.findall(pattern, sql, re.IGNORECASE)
    
    for match in matches:
        schema_name, table_name = match
        # If schema name exists, use table name only
        if table_name:
            table_names.append(table_name)
    
    # Remove duplicates while preserving order
    unique_table_names = []
    for name in table_names:
        if name not in unique_table_names:
            unique_table_names.append(name)
            
    return unique_table_names


def process_json_file(input_file: str, table_names: str, output_file: str, type: str) -> None:
    """
    Process JSON file to extract table names from SQL and add them as a new field.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    """
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if type == "llm":
        with open(table_names, 'r', encoding='utf-8') as f:
            table_names = json.load(f)
        for item_data, item_table_names in zip(data, table_names):
            # Process each item in the JSON array
            for item_data, item_table_names in zip(data, table_names):
                item_data['table_names'] = item_table_names['table_names']
    elif type == "gold_answer":
        for item in data:
            # Extract SQL from the item
            sql = item.get('SQL', '')
            if sql:
                # Extract table names from SQL
                table_names = extract_table_names_from_sql(sql)
                # Add table_names as a new field
                item['table_names'] = table_names

    # Save the updated data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 从标准sql中提取表名或从模型中生成的表名中提取表名，保存为{
#     "question_id": 0,
#     "db_id": "california_schools",
#     "question": "",
#     "evidence": "",
#     "SQL": "",
#     "difficulty": "simple",
#     "table_names": [
#       "frpm"
#     ]
#   }
def main():
    input = "/home/scolcy/work/bird/dev_20240627/dev.json"
    # input = "/home/scolcy/work/bird/train/train.json"
    table_names = "/home/scolcy/work/bird/dev_20240627/schema_linking/schema_linking_table_sco3b_lora_train.json"
    output = "/home/scolcy/work/bird/dev_20240627/dev_gold_extend_table_names.json"
    # output = "/home/scolcy/work/bird/train/train_extend_gold_table_names.json"
    type = "gold_answer" # llm or gold_answer
    process_json_file(input, table_names, output, type)
    print(f"Processing complete. Output saved to {output}")


if __name__ == '__main__':
    main()