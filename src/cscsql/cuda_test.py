import torch

from cscsql.utils.common_utils import CommonUtils

# 检查是否有可用 GPU
print(torch.cuda.is_available())  # 输出 True 表示可用
print(torch.cuda.device_count())  # 输出可用 GPU 数量
print(torch.cuda.get_device_name(0))  # 输出第 0 块 GPU 名称


few_shot_results = CommonUtils.get_few_shot_list()
print(few_shot_results)
