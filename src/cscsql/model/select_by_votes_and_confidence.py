import json
import os
import math


def load_json_data(json_path):
    """加载JSON文件数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sigmoid(x):
    """Sigmoid函数"""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)


def calculate_best_sql(vote_list, sql_list, avg_logprobs, w=0.5):
    """
    计算最优SQL（仅考虑top1和top2）
    :param vote_list: top1和top2的票数列表
    :param sql_list: top1和top2的SQL语句列表
    :param avg_logprobs: top1和top2的平均对数概率（置信度）列表
    :param w: 票数权重（0-1之间）
    :return: 得分最高的SQL语句
    """
    # 提取top1和top2的票数、置信度
    v1, v2 = vote_list[0], vote_list[1]
    c1, c2 = avg_logprobs[0], avg_logprobs[1]
    sql1, sql2 = sql_list[0], sql_list[1]

    # 1. 计算top2内的票数占比（RVR）
    total_vote = v1 + v2
    if total_vote == 0:
        # 极端情况：两者均无票，平等对待
        rvr1, rvr2 = 0.5, 0.5
    else:
        rvr1 = v1 / total_vote
        rvr2 = v2 / total_vote

    # 2. 计算top2内的归一化置信度（RNC）- 使用更平滑的方法
    # 使用softmax-like方法进行归一化，让差异更平滑
    diff = c1 - c2
    # 使用sigmoid函数来平滑差异
    softmax_sum = math.exp(c1) + math.exp(c2)
    rnc1 = math.exp(c1) / softmax_sum
    rnc2 = math.exp(c2) / softmax_sum
    print("RVR:", rvr1, rvr2)
    print("RNC:", rnc1, rnc2)

    # 3. 计算总得分
    score1 = w * rvr1 + (1 - w) * rnc1
    score2 = w * rvr2 + (1 - w) * rnc2

    # 4. 选择得分更高的SQL（若得分相同，优先选top1）
    if score1 >= score2:
        return sql1, score1, score2
    else:
        return sql2, score1, score2


def calculate_best_sql_calibrated(vote_list, sql_list, avg_logprobs, w=0.5):
    """
    使用校准的置信度评分 - 简洁有效版本
    """
    v1, v2 = vote_list[0], vote_list[1]
    c1, c2 = avg_logprobs[0], avg_logprobs[1]
    sql1, sql2 = sql_list[0], sql_list[1]

    # 1. 计算票数比例
    total_votes = v1 + v2
    if total_votes == 0:
        vote_ratio1 = vote_ratio2 = 0.5
    else:
        vote_ratio1 = v1 / total_votes
        vote_ratio2 = v2 / total_votes

    # 2. 置信度归一化（softmax）
    # 由于avg_logprobs通常为负值，先进行偏移
    max_conf = max(c1, c2)
    exp_c1 = math.exp(c1 - max_conf)  # 数值稳定性
    exp_c2 = math.exp(c2 - max_conf)
    conf_sum = exp_c1 + exp_c2
    conf_ratio1 = exp_c1 / conf_sum
    conf_ratio2 = exp_c2 / conf_sum

    # 3. 基于票数差异的动态权重
    if total_votes == 0:
        dynamic_w = 0.0  # 无票数信息时完全依赖置信度
    else:
        vote_diff_ratio = abs(v1 - v2) / total_votes
        # 票数差异越大，权重向票数倾斜
        dynamic_w = min(w + vote_diff_ratio, 1.0)

    # 4. 计算最终得分
    score1 = dynamic_w * vote_ratio1 + (1 - dynamic_w) * conf_ratio1
    score2 = dynamic_w * vote_ratio2 + (1 - dynamic_w) * conf_ratio2

    if score1 >= score2:
        return sql1, score1, score2
    else:
        return sql2, score1, score2

def save_sql_results_to_file(sql_results, input_json_path):
    """将所有SQL结果保存到一个文件中，每行一个SQL语句"""
    # 获取输入JSON文件的目录
    input_dir = os.path.dirname(input_json_path)
    # 构造输出文件路径
    output_file_path = os.path.join(input_dir, 'selected_sql_results.sql')

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for sql in sql_results:
            f.write(sql + '\n')

    print(f"已将所有SQL结果保存到文件: {output_file_path}")


def main(json_path, w=0.5):
    """主函数：串联流程"""
    # 1. 加载JSON数据
    data_list = load_json_data(json_path)
    print(f"成功加载 {len(data_list)} 个问题的数据")

    # 存储所有最优SQL结果
    best_sql_results = []

    # 2. 逐个处理问题
    for data in data_list:
        question_id = data['question_id']
        vote_list = data['vote_list']
        sql_list = data['sql']
        
        # 检查是否存在avg_logprobs字段
        if 'avg_logprobs' in data:
            avg_logprobs = data['avg_logprobs']
        else:
            print(f"警告: 问题 {question_id} 缺少 'avg_logprobs' 字段，使用默认置信度值")
            # 如果没有avg_logprobs，则使用默认值（例如全为0）
            avg_logprobs = [0.0] * len(sql_list)
        
        best_sql = None

        # 确保top1和top2存在
        if len(vote_list) < 2 or len(sql_list) < 2 or len(avg_logprobs) < 2:
            best_sql = sql_list[0] if sql_list else ""
        else:
            # 3. 计算最优SQL
            try:
                best_sql, score1, score2 = calculate_best_sql(
                    vote_list=vote_list,
                    sql_list=sql_list,
                    avg_logprobs=avg_logprobs,
                    w=w
                )
                # 打印计算过程（可选，用于调试）
                print(f"\n问题 {question_id} 计算结果：")
                print(f"top1票数：{vote_list[0]}，置信度：{avg_logprobs[0]}，得分：{score1:.4f}")
                print(f"top2票数：{vote_list[1]}，置信度：{avg_logprobs[1]}，得分：{score2:.4f}")
                print(f"最优SQL为：{best_sql[:50]}...")  # 只显示前50字符
            except Exception as e:
                print(f"问题 {question_id} 计算过程中出现错误: {e}，使用第一个SQL作为默认结果")
                best_sql = sql_list[0] if sql_list else ""
        
        # 添加到结果列表
        best_sql_results.append(best_sql)

    # 4. 保存所有结果到一个文件
    save_sql_results_to_file(best_sql_results, json_path)


if __name__ == '__main__':
    # 配置参数
    JSON_FILE_PATH = '/mnt/d/text2SQLProject/csc_sql/src/cscsql/model/outputs/20251025_143927/sampling_think_sql_generate_pred_major_top2_sqls.json'  # 输入JSON文件路径
    VOTE_WEIGHT = 0.3# 票数权重（0-1之间，可调整）

    # 执行主函数
    main(json_path=JSON_FILE_PATH, w=VOTE_WEIGHT)