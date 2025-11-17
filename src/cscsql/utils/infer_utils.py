import json
import random
import sys
import os
import sqlite3
from collections import defaultdict
from copy import deepcopy
from typing import Union, List, Dict

from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import multiprocessing as mp

from cscsql.utils.common_utils import CommonUtils, read_packed_sql
from cscsql.utils.file_utils import FileUtils
from cscsql.utils.logger_utils import logger

execution_results = None
gold_execution_results = None


def build_stop_token_ids(pretrained_model_name_or_path: str):
    if "Qwen2.5-" in pretrained_model_name_or_path \
            or "XiYanSQL" in pretrained_model_name_or_path \
            or "Qwen" in pretrained_model_name_or_path \
            or "Qwen2___5" in pretrained_model_name_or_path:
        stop_token_ids = [151645]  # 151645 is the token id of  (end of turn token in Qwen2.5)
    elif "deepseek-coder-" in pretrained_model_name_or_path:
        stop_token_ids = [32021]
    elif "DeepSeek-Coder-V2" in pretrained_model_name_or_path:
        stop_token_ids = [100001]
    elif "OpenCoder-" in pretrained_model_name_or_path:
        stop_token_ids = [96539]
    elif "Meta-Llama-" in pretrained_model_name_or_path:
        stop_token_ids = [128009, 128001]
    elif "granite-" in pretrained_model_name_or_path:
        stop_token_ids = [0]  # <|end_of_text|> is the end token of granite-3.1 and granite-code
    elif "starcoder2-" in pretrained_model_name_or_path:
        stop_token_ids = [0]  # <|end_of_text|> is the end token of starcoder2
    elif "Codestral-" in pretrained_model_name_or_path:
        stop_token_ids = [2]
    elif "Mixtral-" in pretrained_model_name_or_path:
        stop_token_ids = [2]
    elif "OmniSQL-" in pretrained_model_name_or_path:
        stop_token_ids = [151645]  # OmniSQL uses the same tokenizer as Qwen2.5
    elif "gemma-3" in pretrained_model_name_or_path:
        stop_token_ids = [1, 106]
    else:
        print("Use empty stop tokens.")
        stop_token_ids = None
    # print("stop_token_ids:", stop_token_ids)

    return stop_token_ids


def build_execute_sql_result(gen_sqls: str, db_path: str):
    predict_sql_results = None
    if gen_sqls is None or gen_sqls in ["none"]:
        return predict_sql_results

    from cscsql.utils.logger_utils import logger
    logger.info(f"read gen sql file: {gen_sqls} ")
    predict_sql_results = CommonUtils.execute_batch_sql_to_result(gen_sqls,
                                                                  db_root=db_path)

    return predict_sql_results


def build_selection_vote_execute_sql_result(selection_vote: Union[str, int],
                                            db_path: str,
                                            is_train=False,
                                            prompt_mode="merge"):
    predict_sql_results = None
    all_predict_results = []
    if selection_vote is None or selection_vote in ["none"]:
        return predict_sql_results, all_predict_results

    logger.info(f"read selection vote sql file: {prompt_mode} - {selection_vote} ")
    predict_sql_results, all_predict_results = CommonUtils.execute_batch_vote_sql_to_result(selection_vote,
                                                                                            db_root=db_path,
                                                                                            is_train=is_train,
                                                                                            prompt_mode=prompt_mode)

    return predict_sql_results, all_predict_results


def execute_sql(data_idx, db_file, sql):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        execution_res = frozenset(execution_res)  # make set hashable
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, execution_res, 1

    except:
        conn.rollback()
        conn.close()
        return data_idx, db_file, sql, None, 0


def execute_sql_wrapper(data_idx, db_file, sql, timeout):
    try:
        res = func_timeout(timeout, execute_sql, args=(data_idx, db_file, sql))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = (data_idx, db_file, sql, None, 0)
    except Exception as e:
        res = (data_idx, db_file, sql, None, 0)

    return res


def execute_callback_execute_sqls(result):
    data_idx, db_file, sql, query_result, valid = result

    execution_results.append(
        {
            "data_idx": data_idx,
            "db_file": db_file,
            "sql": sql,
            "query_result": query_result,
            "valid": valid
        }
    )


def execute_callback_execute_sqls_gold(result):
    data_idx, db_file, sql, query_result, valid = result

    gold_execution_results.append(
        {
            "data_idx": data_idx,
            "db_file": db_file,
            "sql": sql,
            "query_result": query_result,
            "valid": valid
        }
    )


def execute_sqls_parallel(db_files, sqls, num_cpus=1, timeout=1):
    # 如果SQL数量超过条，则分批执行以减轻CPU压力
    max_batch_size = 500
    total_batches = (len(sqls) + max_batch_size - 1) // max_batch_size  # 计算总批次数
    if len(sqls) <= max_batch_size:
        # SQL数量不超过15000条，直接执行
        print(f"SQL数量({len(sqls)})不超过{max_batch_size}条，直接执行...")
        _execute_sqls_batch(db_files, sqls, 0, num_cpus, timeout)  # 从索引0开始
    else:
        # SQL数量超过条，分批执行
        print(f"SQL数量({len(sqls)})超过{max_batch_size}条，将分批执行，共{total_batches}批...")
        batch_count = 0
        for i in range(0, len(sqls), max_batch_size):
            batch_db_files = db_files[i:i + max_batch_size]
            batch_sqls = sqls[i:i + max_batch_size]
            current_batch = i//max_batch_size + 1
            base_idx = i  # 当前批次的起始索引
            print(f"执行第{current_batch}批次，包含{len(batch_sqls)}条SQL，起始索引: {base_idx}")
            try:
                _execute_sqls_batch(batch_db_files, batch_sqls, base_idx, num_cpus, timeout)
                print(f"第{current_batch}批次执行完成")
                batch_count += 1
                print(f"进度: {batch_count}/{total_batches} 批次完成")
            except Exception as e:
                print(f"执行第{current_batch}批次时发生异常: {e}")
                raise e  # 重新抛出异常，让调用者知道
        print(f"所有批次执行完成，共执行了{batch_count}个批次")


def _execute_sqls_batch(db_files, sqls, base_idx, num_cpus=1, timeout=1):
    print(f"创建进程池，使用{num_cpus}个CPU核心处理{len(sqls)}条SQL，起始索引: {base_idx}")
    pool = mp.Pool(processes=num_cpus)
    results = []
    for idx, db_file, sql in zip(list(range(len(sqls))), db_files, sqls):
        data_idx = base_idx + idx  # 使用全局索引而不是局部索引
        result = pool.apply_async(execute_sql_wrapper, args=(data_idx, db_file, sql, timeout),
                         callback=execute_callback_execute_sqls)
        results.append(result)
    
    print(f"提交了{len(results)}个异步任务，开始等待任务完成...")
    # 等待所有任务完成，并检查是否有异常
    for i, result in enumerate(results):
        try:
            result.get(timeout=timeout + 5)  # 等待结果，超时时间比SQL执行超时稍长
        except Exception as e:
            print(f"警告：第{i}个SQL任务执行异常: {e}")
    
    pool.close()
    pool.join()
    print(f"进程池任务完成，处理了{len(sqls)}条SQL")


def execute_gold_sqls_parallel(db_files, sqls, num_cpus=1, timeout=1):
    pool = mp.Pool(processes=num_cpus)
    for data_idx, db_file, sql in zip(list(range(len(sqls))), db_files, sqls):
        pool.apply_async(execute_sql_wrapper, args=(data_idx, db_file, sql, timeout),
                         callback=execute_callback_execute_sqls_gold)
    pool.close()
    pool.join()


def major_voting2(db_files, pred_sqls, sampling_num,
                  ground_truth_sqls, gold_db_files,
                  return_random_one_when_all_errors=True,
                  num_cpus=10, timeout=30, 
                  avg_logprobs=None):
    global execution_results
    global gold_execution_results
    mj_pred_correctness_list = []
    mj_top2_pred_correctness_list = []
    execution_results = []
    # execute all sampled SQL queries to obtain their execution results
    print(f"开始执行SQL，总共{len(pred_sqls)}条SQL语句")
    execute_sqls_parallel(db_files, pred_sqls, num_cpus=num_cpus, timeout=timeout)
    print("sql exe over")
    execution_results = sorted(execution_results, key=lambda x: x['data_idx'])
    print("len(execution_results):", len(execution_results))

    gold_execution_results = []
    if ground_truth_sqls:
        execute_gold_sqls_parallel(gold_db_files, ground_truth_sqls, num_cpus=num_cpus, timeout=timeout)
        gold_execution_results = sorted(gold_execution_results, key=lambda x: x['data_idx'])
    print("len(gold_execution_results):", len(gold_execution_results))

    # perform major voting
    correctness_list = []
    upper_correctness_list = []
    top2_correctness_list = []
    question_idx = 0
    data_idx = 0
    for result_idx in range(0, len(execution_results), sampling_num):
        major_voting_counting = dict()
        execution_results_of_one_sample = execution_results[result_idx: result_idx + sampling_num]

        gold_query_result = None
        if ground_truth_sqls and gold_execution_results:
            gold_execution_results_one = gold_execution_results[question_idx]
            gold_query_result = gold_execution_results_one["query_result"]
        question_idx += 1

        # if no predicted SQLs are valid
        if sum([res["valid"] for res in execution_results_of_one_sample]) == 0:
            if return_random_one_when_all_errors:
                mj_pred_sql = random.choice(execution_results_of_one_sample)["sql"]  # select a random one to return
            else:
                mj_pred_sql = "Error SQL"

            mj_item = {
                'question_id': question_idx,
                "votes": 1,
                'correctness': 0,
                "sql": mj_pred_sql
            }
            mj_pred_correctness_list.append(mj_item)
            mj_top2_pred_correctness_list.append(mj_item)
            correctness_list.append(0)

            upper_item = {
                'question_id': question_idx,
                'correctness': 0,
                'sql': mj_pred_sql
            }
            upper_correctness_list.append(upper_item)
            top2_correctness_list.append({
                'question_id': question_idx,
                'sampling_num': sampling_num,
                'correctness': 0,
                'correctness_list': [0],
                'vote_list': [0],
                'sql': [mj_pred_sql]
            })
            
            # 处理logprobs
            if avg_logprobs is not None:
                top2_correctness_list[-1]['avg_logprobs'] = [0.0]  # 默认值
            
            data_idx += sampling_num
            continue

        current_correctness_list = []
        current_correctness_sql = execution_results_of_one_sample[0]["sql"]
        current_logprobs = []
        for idx_in_sample, res in enumerate(execution_results_of_one_sample):
            query_result = res['query_result']
            valid = res['valid']
            current_is_correct = 0
            if valid == 1:  # skip invalid SQLs
                if gold_query_result and query_result == gold_query_result:
                    current_is_correct = 1
                    current_correctness_sql = res["sql"]
                current_correctness_list.append(current_is_correct)

                if query_result in major_voting_counting:
                    major_voting_counting[query_result]["votes"] += 1
                    # 收集相同query_result的logprobs
                    if avg_logprobs is not None and data_idx + idx_in_sample < len(avg_logprobs):
                        major_voting_counting[query_result]["logprobs"].append(avg_logprobs[data_idx + idx_in_sample])
                else:
                    # 获取对应的logprob
                    logprobs_list = []
                    if avg_logprobs is not None and data_idx + idx_in_sample < len(avg_logprobs):
                        logprobs_list = [avg_logprobs[data_idx + idx_in_sample]]
                    
                    major_voting_counting[query_result] = {
                        'question_id': question_idx,
                        "votes": 1,
                        'correctness': current_is_correct,
                        "sql": res["sql"],
                        "logprobs": logprobs_list
                    }

            else:
                current_correctness_list.append(0)

        upper_item = {
            'question_id': question_idx,
            'correctness': 0,
            'sql': current_correctness_sql
        }
        if 1 in current_correctness_list:
            upper_item['correctness'] = 1
        upper_correctness_list.append(upper_item)

        # find the SQL with the max votes
        all_votes = [vote["votes"] for vote in major_voting_counting.values()]
        all_votes.sort(reverse=True)
        top1_vote = all_votes[0]
        top2_vote = top1_vote
        if len(all_votes) > 1:
            top2_vote = all_votes[1]

        # 计算每个query_result的平均logprobs
        for query_result, vote_info in major_voting_counting.items():
            if vote_info["logprobs"]:
                vote_info["avg_logprob"] = sum(vote_info["logprobs"]) / len(vote_info["logprobs"])
            else:
                vote_info["avg_logprob"] = 0.0

        major_vote = max(major_voting_counting.values(), key=lambda x: x["votes"])
        top1_key = None
        top2_key = None
        for k, v in major_voting_counting.items():
            vote = v["votes"]
            if vote == major_vote["votes"]:
                top1_key = k
            if vote == top2_vote:
                top2_key = k
            if top1_key is not None and top2_key is not None:
                break

        mj_item = major_voting_counting[top1_key]
        mj_pred_correctness_list.append(mj_item)
        mj_top2_item = major_voting_counting[top2_key]
        mj_top2_pred_correctness_list.append(mj_top2_item)

        top2_item = {
            'question_id': question_idx,
            'sampling_num': sampling_num,
            'correctness': 1 if mj_item['correctness'] == 1 or mj_top2_item['correctness'] == 1 else 0,
            "correctness_list": [mj_item['correctness'], mj_top2_item['correctness']] if top1_key != top2_key else [
                mj_item['correctness']],
            'vote_list': [top1_vote, top2_vote] if top1_key != top2_key else [top1_vote],
            'sql': [mj_item["sql"], mj_top2_item["sql"]] if top1_key != top2_key else [mj_item["sql"]],
            'avg_logprobs': [mj_item["avg_logprob"], mj_top2_item["avg_logprob"]] if top1_key != top2_key else [mj_item["avg_logprob"]]
        }

        top2_correctness_list.append(top2_item)
        data_idx += sampling_num

    return mj_pred_correctness_list, upper_correctness_list, top2_correctness_list


def calc_nl2sql_result(evaluation_scores: List, gold: List[Dict]):
    execution_accuracy = sum(evaluation_scores) / len(evaluation_scores)

    eval_sep_acc = {}

    eval_data_config = defaultdict(int)
    eval_tp_config = defaultdict(int)

    tp_id = []
    if gold and len(gold) == len(evaluation_scores):
        for index, item in enumerate(gold):
            difficulty = item.get("difficulty", "simple")
            eval_data_config[difficulty] += 1

            predict = evaluation_scores[index]
            if predict != 1:
                continue

            eval_tp_config[difficulty] += 1
            tp_id.append(index + 1)

        eval_sep_acc = {}
        for k, v in eval_data_config.items():
            tp = eval_tp_config.get(k, 0)
            eval_sep_acc[k] = tp / v if v > 0 else 0

    metric = {
        "easy": eval_sep_acc.get("simple", 0),
        "easy_total": eval_data_config.get("simple", 0),
        "medium": eval_sep_acc.get("moderate", 0),
        "medium_total": eval_data_config.get("moderate", 0),
        "hard": eval_sep_acc.get("challenging", 0),
        "hard_total": eval_data_config.get("moderate", 0),
        "extra": 0,
        "extra_total": 0,
        "all": execution_accuracy,
        "all_total": len(evaluation_scores),
        "acc": execution_accuracy,
        "tp_id": tp_id,
    }

    metric.update(eval_sep_acc)

    return metric


def run_eval_major_vote(gold_file, pred_file, db_path,
                        num_cpus=8, timeout=30, pred_sql_key="pred_sqls",
                        config: Dict = None):
    pred_results = FileUtils.load_json(pred_file)

    dev_file = db_path.replace("_databases", ".json")
    dev_data = FileUtils.load_json(dev_file)

    if gold_file and len(gold_file) > 10:
        ground_truth_sqls, gold_dbs = read_packed_sql(gold_file, db_root=db_path)
    else:
        ground_truth_sqls = None
        gold_dbs = None

    sampling_num = len(pred_results[0][pred_sql_key])
    print("sampling_num:", sampling_num)

    db_files = []
    gold_db_files = []
    pred_sqls = []
    avg_logprobs = []  # 收集所有avg_logprobs
    for pred_data in pred_results:
        db_id = pred_data["db_id"]
        db_file_path = os.path.join(db_path, db_id, db_id + ".sqlite")
        db_files.extend([db_file_path] * sampling_num)
        gold_db_files.append(db_file_path)

        pred_sqls.extend(pred_data[pred_sql_key])
        # 收集对应的avg_logprobs
        if "avg_logprobs" in pred_data:
            avg_logprobs.extend(pred_data["avg_logprobs"])
    
    # 如果没有logprobs数据，则设为None
    if not avg_logprobs:
        avg_logprobs = None
        
    assert len(pred_sqls) == len(db_files)

    (mj_pred_correctness_list, upper_correctness_list, top2_correctness_list) = major_voting2(db_files,
                                                                                              pred_sqls,
                                                                                              sampling_num=sampling_num,
                                                                                              ground_truth_sqls=ground_truth_sqls,
                                                                                              gold_db_files=gold_db_files,
                                                                                              num_cpus=num_cpus,
                                                                                              timeout=timeout,
                                                                                              avg_logprobs=avg_logprobs)
    # save the (major-voting) predicted SQL so we can check it out later
    mj_pred_sqls = [item['sql'] for item in mj_pred_correctness_list]
    save_file = pred_file[:-5] + "_pred_major_voting_sqls.json"
    FileUtils.dump_json(save_file, mj_pred_sqls)
    mj_predict_file = save_file.replace(".json", ".sql")
    mj_pred_sqls2 = [item['sql'].replace("\n", " ") for item in mj_pred_correctness_list]
    FileUtils.save_to_text(mj_predict_file, "\n".join(mj_pred_sqls2))

    save_mj_top2_file = save_file.replace("major_voting_sqls.json", "major_top2_sqls.json")
    FileUtils.dump_json(save_mj_top2_file, top2_correctness_list)

    if ground_truth_sqls is None:
        print(f"ground_truth_sqls is None, only return save file: {save_file}")
        return save_file, mj_pred_sqls, None, save_mj_top2_file

    evaluation_scores = [res["correctness"] for res in mj_pred_correctness_list]
    execution_accuracy = sum(evaluation_scores) / len(evaluation_scores)

    mj_metric = calc_nl2sql_result(evaluation_scores, dev_data)
    print("EX Accuracy (major voting):", execution_accuracy)

    pass_at_k_scores = [res["correctness"] for res in upper_correctness_list]
    pass_at_k_metric = calc_nl2sql_result(pass_at_k_scores, dev_data)
    print(f"EX Accuracy (pass@{sampling_num}):", sum(pass_at_k_scores) / len(pass_at_k_scores))

    mj_pass_at_2_scores = [res["correctness"] for res in top2_correctness_list]
    mj_pass_at_2_metric = calc_nl2sql_result(mj_pass_at_2_scores, dev_data)
    print(f"EX Accuracy (major_top2@{sampling_num}):", sum(mj_pass_at_2_scores) / len(mj_pass_at_2_scores))

    mj_metric1 = deepcopy(mj_metric)
    mj_metric1.pop("tp_id")

    pass_at_k_metric1 = deepcopy(pass_at_k_metric)
    pass_at_k_metric1.pop("tp_id")

    mj_pass_at_2_metric1 = deepcopy(mj_pass_at_2_metric)
    mj_pass_at_2_metric1.pop("tp_id")
    metric = {
        "config": config,
        "mj_predict_file": mj_predict_file,
        "mj_metric": mj_metric1,
        "pass_at_k_metric": pass_at_k_metric1,
        "mj_pass_at_2_metric": mj_pass_at_2_metric1,
    }

    metric_save_file = pred_file[:-5] + "_metric.json"
    FileUtils.dump_json(metric_save_file, metric)

    return mj_metric, mj_pred_sqls, pass_at_k_metric, mj_pass_at_2_metric


def run_eval_major_vote_table(gold_file, pred_file, db_path,
                              pred_sql_key="pred_sqls",
                              config: Dict = None):
    pred_results = FileUtils.load_json(pred_file)

    if len(pred_results) == 0:
        return None

    dev_file = db_path.replace("_databases", ".json")
    dev_data = FileUtils.load_json(dev_file)

    if gold_file and len(gold_file) > 10:
        ground_truth_sqls, gold_dbs = read_packed_sql(gold_file, db_root=db_path)
    else:
        ground_truth_sqls = None
        gold_dbs = None

    sampling_num = len(pred_results[0][pred_sql_key])
    print("eval link table, sampling_num:", sampling_num)

    total_samples = len(pred_results)
    total_accuracy = 0
    filtered_accuracy = 0
    total_precision = 0
    total_recall = 0

    results = []
    tp_ids = []
    for index, item in enumerate(pred_results):
        pred_sqls = item[pred_sql_key]
        reference_tables = item.get("output", [])
        raw_predict_tables = []
        for gen in pred_sqls:
            raw_predict_tables.extend(gen)
        predicted_tables = list(set(raw_predict_tables))

        # Convert to lowercase and strip whitespace for comparison
        predicted_tables = [x.lower().strip() for x in predicted_tables]
        reference_tables = [x.lower().strip() for x in reference_tables]

        # Calculate accuracy
        acc_flag = False
        if set(predicted_tables) == set(reference_tables):
            total_accuracy += 1
            acc_flag = True

        # Calculate precision and recall
        true_positives = len(set(predicted_tables) & set(reference_tables))
        false_positives = len(set(predicted_tables) - set(reference_tables))
        false_negatives = len(set(reference_tables) - set(predicted_tables))

        filter_acc_flag = False
        if true_positives == len(reference_tables):
            filtered_accuracy += 1
            filter_acc_flag = True
            tp_ids.append(index)

        precision = 0
        recall = 0
        if len(predicted_tables) > 0:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

        total_precision += precision
        total_recall += recall

        one_predict = {
            "id": index,
            "predicted_tables": predicted_tables,
            "reference_tables": reference_tables,
            "acc_flag": acc_flag,
            "filter_acc_flag": filter_acc_flag,
            "precision": precision,
            "recall": recall,
        }
        results.append(one_predict)

    # Calculate average precision and recall
    avg_precision = total_precision / total_samples
    avg_recall = total_recall / total_samples
    avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

    # Calculate total accuracy
    accuracy = total_accuracy / total_samples
    filtered_accuracy = filtered_accuracy / total_samples

    predict_file = str(pred_file).replace(".json", "_sep_metric.json")

    table_metric = {
        "all_total": total_samples,
        "acc": accuracy,
        "acc_filter": filtered_accuracy,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "metric_file": predict_file
    }

    print(f"link table result total=[{total_samples}]:\nacc: {accuracy:.3f} - acc_filter:{filtered_accuracy:.3f} "
          f"avg_precision:{avg_precision:.3f} - avg_recall:{avg_recall:.3f} "
          f"avg_f1:{avg_f1:.3f} \n"
          f"- metric_file: {predict_file}")

    FileUtils.dump_json(predict_file, results)

    metric = {
        "config": config,
        "metric_file": predict_file,
        "table_metric": table_metric,
    }

    metric_save_file = str(pred_file).replace(".json", "_metric.json")
    FileUtils.dump_json(metric_save_file, metric)

    return metric
