import json
import os
from typing import List, Dict

from cscsql.utils.extract_table_names import extract_table_names_from_sql
from cscsql.utils.file_utils import FileUtils
from cscsql.utils.llm_table_extractor import create_table_extraction_prompt, parse_llm_table_response


def save_sft_data(
    output_path: str,
    instructions: List[str],
    outputs: List[str],
    systems: List[str] = None
) -> None:
    """
    将系统提示词和用户提示词保存为SFT格式的JSON文件
    
    Args:
        output_path (str): 输出文件路径
        instructions (List[str]): 用户提示词列表
        outputs (List[str]): 模型回答列表
        systems (List[str], optional): 系统提示词列表，默认为None
    """

    # 构造SFT数据格式
    sft_data = []
    for i in range(len(instructions)):
        item = {
            "instruction": instructions[i],
            "input": "",
            "output": outputs[i],
            "system": systems[i] if systems else ""
        }
        sft_data.append(item)
    
    # 保存到JSON文件
    FileUtils.dump_json(output_path, sft_data)
    print(f"SFT数据已保存到: {output_path}")


def convert_inference_results_to_sft_data(
    input_file: str,
    output_file: str
) -> None:
    """
    将推理结果转换为SFT格式数据
    
    Args:
        input_file (str): 输入的推理结果文件路径（JSON格式）
        output_file (str): 输出的SFT数据文件路径
    """
    # 读取推理结果
    inference_results = FileUtils.load_json(input_file)
    
    instructions = []
    outputs = []
    systems = []
    
    # 解析推理结果
    for item in inference_results:
        # 假设input_seq是用户提示词
        instructions.append(item.get("input_seq", ""))

        # 选择最佳的模型输出（这里选择第一个）
        responses = item.get("responses", [])
        pred_sqls = item.get("pred_sqls", [])
        
        # 使用预测的SQL作为输出，如果不存在则使用原始响应
        output = ""
        if pred_sqls and len(pred_sqls) > 0:
            output = pred_sqls[0]
        elif responses and len(responses) > 0:
            output = responses[0]
        else:
            output = ""
            
        outputs.append(output)
        
        # 添加默认系统提示词（可根据实际情况修改）
        systems.append("You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>...</think>\n<answer>\n...\n</answer>")
    
    # 保存为SFT格式
    save_sft_data(output_file, instructions, outputs, systems)
def create_output(table_names: List[str]):
    return f"""'''json {{"table_names": {json.dumps(table_names)}}}'''"""

def create_stf_data(input_data):
    instructions = []
    systems = []
    outputs = []
    system_prompt = """You are a helpful AI Assistant that provides well-reasoned and detailed responses.Respond in the following format with no explaination: ```json{"table_names": ["table1", "table2", "table3"]}```"""
    for idx, item in enumerate(input_data):
        # if idx > 5:
        #      break
        input_seq = item.get("input_seq", "")
        output_seq = item.get("output_seq", "")
        # 从sql中提取表名
        table_names = extract_table_names_from_sql(output_seq)
        # 创建json标准输出
        output = create_output(table_names)
        # 创建提示
        prompt = create_table_extraction_prompt(input_seq)

        instructions.append(prompt)
        systems.append(system_prompt)
        outputs.append(output)



    return instructions, systems, outputs


if __name__ == "__main__":
    # input = "/home/scolcy/work/bird/dev_20240627/dev_bird.json"
    input = "/home/scolcy/work/bird/train/train_bird_prompt_full_tables.json"
    with open(input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    instructions, systems, outputs = create_stf_data(input_data)

    save_sft_data(
        output_path="/home/scolcy/work/bird/dev_20240627/stf/sft_train.json",
        instructions=instructions,
        outputs=outputs,
        systems=systems
    )

    # 2. 转换现有推理结果文件
    # convert_inference_results_to_sft_data(
    #     input_file="inference_results.json",
    #     output_file="sft_data_from_inference.json"
    # )