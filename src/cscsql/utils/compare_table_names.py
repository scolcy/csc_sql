import json
from typing import Dict, List, Tuple


def compare_table_names(ground_truth: List[Dict], prediction: List[Dict]) -> Dict[str, float]:
    """
    Compare table names between ground truth and prediction.
    
    Args:
        ground_truth: List of dictionaries containing question_id and table_names (ground truth)
        prediction: List of dictionaries containing question_id and table_names (prediction)
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Convert lists to dictionaries for easier lookup
    # Handle  case where table_names might be None
    gt_dict = {}
    for item in ground_truth:
        table_names = item.get('table_names', []) or []
        gt_dict[item['question_id']] = set(table_names)
    
    pred_dict = {}
    for item in prediction:
        table_names = item.get('table_names', []) or []
        pred_dict[item['question_id']] = set(table_names)
    
    total_questions = len(gt_dict)
    correct_count = 0
    missing_count = 0  # 只漏预测问题数
    missing_ids = []
    extra_count = 0  # 只多预测问题数
    extra_ids = []
    missing_ids = []
    both_count = 0  # 同时存在漏预测和多预测的问题数
    both_ids = []
    
    # 计算完全匹配的数量
    for question_id, gt_tables in gt_dict.items():
        if question_id in pred_dict:
            pred_tables = pred_dict[question_id]
            
            # 完全正确预测的数量
            if gt_tables == pred_tables:
                correct_count += 1
                
            # 漏预测的数量 (在真实答案中但不在预测中)
            missing_tables = gt_tables - pred_tables

            
            # 多预测的数量 (在预测中但不在真实答案中)
            extra_tables = pred_tables - gt_tables

            # 根据题目要求进行统计
            if missing_tables and extra_tables:
                # 同时存在漏预测和多预测
                both_count += 1
                both_ids.append(question_id)
            elif missing_tables:
                # 只有漏预测
                missing_count += 1
                missing_ids.append(question_id)
            elif extra_tables:
                # 只有多预测
                extra_count += 1
                extra_ids.append(question_id)

    
    # 计算准确率
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    return {
        "total_questions": total_questions,
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "missing_tables": missing_count,
        "extra_tables": extra_count,
        "miss and extra": both_count,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "both_ids": both_ids
    }


def load_json_data(file_path: str) -> List[Dict]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":

    
    ground_truth_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_extend_table_names.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_Qwen2.5-3B-Instruct_extend_table_names.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_Qwen3-4B-Instruct-2507_lora_schema_linking_extend_table_names3.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_Qwen3-4B-Instruct-2507-unsloth-bnb-4bit_schema_linking_extend_table_names.json"
    prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_Qwen3-4B-Instruct-2507-unsloth-bnb-4bit_lora16_schema_linking_extend_table_names.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_Qwen3-4B-Instruct-2507_lora_r12_schema_linking_extend_table_names.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_Qwen3-4B-Instruct-2507_extend_table_names.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_qwen_max_extend_table_names.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_qwen_plus_extend_table_names.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev_XiYanSQL-QwenCoder-3B-2504_extend_table_names.json"
    # prediction_file = "/home/scolcy/work/bird/dev_20240627/dev.json extend tablenames/dev_gold_extend_table_names.json"

    # Load data
    ground_truth = load_json_data(ground_truth_file)

    prediction = load_json_data(prediction_file)
    
    # Compare
    results = compare_table_names(ground_truth, prediction)
    
    # Print results
    print("Table Names Comparison Results:")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Missing Tables: {results['missing_tables']}")
    print(f"Extra Tables: {results['extra_tables']}")
    print(f"Miss and Extra Tables: {results['miss and extra']}")
    print(f"Missing IDs: {results['missing_ids']}")
    print(f"Extra IDs: {results['extra_ids']}")
    print(f"Both IDs: {results['both_ids']}")