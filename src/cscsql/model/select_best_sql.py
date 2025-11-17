import argparse
import json
from typing import List, Dict, Any


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from a file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Dict[str, Any]]: Loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(file_path: str, data: List[Dict[str, Any]]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        file_path (str): Path to the output JSON file
        data (List[Dict[str, Any]]): Data to save
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_sql_file(file_path: str, sql_list: List[str]) -> None:
    """
    Save SQL statements to a .sql file, one SQL per line.
    
    Args:
        file_path (str): Path to the output SQL file
        sql_list (List[str]): List of SQL statements to save
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for sql in sql_list:
            # Clean the SQL by removing newlines and extra spaces
            clean_sql = sql.replace('\n', ' ').replace('\r', '').strip()
            f.write(clean_sql + '\n')


def select_best_sql_by_logprob(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Select the SQL with the highest cumulative log probability for each question.
    
    Args:
        predictions (List[Dict[str, Any]]): List of prediction results with multiple SQLs and their logprobs
        
    Returns:
        List[Dict[str, Any]]: List of best SQL predictions for each question
    """
    best_predictions = []
    
    for item in predictions:
        # Get the SQLs and their cumulative log probabilities
        pred_sqls = item.get('pred_sqls', [])
        avg_logprobs = item.get('avg_logprobs', [])
        
        if not pred_sqls or not avg_logprobs:
            # If no predictions or logprobs, use the first SQL if available
            best_sql = pred_sqls[0] if pred_sqls else ""
            best_logprob = avg_logprobs[0] if avg_logprobs else float('-inf')
        else:
            # Find the SQL with the highest cumulative log probability
            best_index = 0
            best_logprob = avg_logprobs[0]
            
            for i, logprob in enumerate(avg_logprobs):
                if logprob > best_logprob:
                    best_logprob = logprob
                    best_index = i
                    
            best_sql = pred_sqls[best_index]
        
        # Create a new item with only the best SQL
        best_item = {
            'id': item.get('id', 0),
            'db_id': item.get('db_id', ''),
            'input_seq': item.get('input_seq', ''),
            'best_sql': best_sql,
            'best_logprob': best_logprob
        }
        
        # Copy other fields that might be needed
        if 'output_seq' in item:
            best_item['output_seq'] = item['output_seq']
            
        best_predictions.append(best_item)
    
    return best_predictions


def main():
    # Hardcoded file paths
    input_file = "/mnt/d/text2SQLProject/csc_sql/src/cscsql/model/outputs/20251025_143927/sampling_think_sql_generate.json"
    output_file = "/mnt/d/text2SQLProject/csc_sql/src/cscsql/model/outputs/20251025_143927/best_logprob.json"
    sql_file = "/mnt/d/text2SQLProject/csc_sql/src/cscsql/model/outputs/20251025_143927/best_logprob.sql"
    
    # Load predictions
    print(f"Loading predictions from {input_file}")
    predictions = load_json(input_file)
    
    # Select best SQL for each prediction
    print("Selecting best SQL for each question based on cumulative log probabilities")
    best_predictions = select_best_sql_by_logprob(predictions)
    
    # Save results as JSON
    # print(f"Saving best predictions to {output_file}")
    # save_json(output_file, best_predictions)
    
    # Save SQLs to a .sql file
    print(f"Saving SQL statements to {sql_file}")
    best_sqls = [item['best_sql'] for item in best_predictions]
    save_sql_file(sql_file, best_sqls)
    
    print(f"Processed {len(best_predictions)} predictions")
    print("Done!")


if __name__ == '__main__':
    main()