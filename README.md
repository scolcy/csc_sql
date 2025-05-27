# CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning

### Important Links

📖[Arxiv Paper](https://arxiv.org/abs/2505.1327) |
🤗[HuggingFace](https://huggingface.co/collections/cycloneboy/csc-sql-6835c4a52da10c54bbe14f8e) |
🤖[ModelScope](https://modelscope.cn/collections/CSC-SQL-8542177708b643) |

## News
+ `May 27, 2025`: The CSC-SQL 32B model achieved an Execution Accuracy (EX) of 73.67% on the BIRD test set, while the 7B model attained an EX of 71.72%, surpassing all other known methods based on open-source models.
+ `May 25, 2025`: Release model and inference code
+ `May 19, 2025`: Publish the paper to arxiv

## Introduction

> Large language models (LLMs) have demonstrated strong capabilities in translating natural language questions about
> relational databases into SQL queries. In particular, test-time scaling techniques such as Self-Consistency and
> Self-Correction can enhance SQL generation accuracy by increasing computational effort during inference. However,
> these
> methods have notable limitations: Self-Consistency may select suboptimal outputs despite majority votes, while
> Self-Correction typically addresses only syntactic errors. To leverage the strengths of both approaches, we propose
> CSC-SQL, a novel method that integrates Self-Consistency and Self-Correction. CSC-SQL selects the two most frequently
> occurring outputs from parallel sampling and feeds them into a merge revision model for correction. Additionally, we
> employ the Group Relative Policy Optimization (GRPO) algorithm to fine-tune both the SQL generation and revision
> models
> via reinforcement learning, significantly enhancing output quality. Experimental results confirm the effectiveness and
> generalizability of CSC-SQL. On the BIRD development set, our 3B model achieves 65.28% execution accuracy, while the
> 7B
> model achieves 69.19%. The code will be open sourced at https://github.com/CycloneBoy/csc_sql.

![csc_sql_framework](data/image/csc_sql_framework.png)


## Main Results

![csc_sql_result](data/image/csc_sql_result.png)

## Model

| **Model and Dataset**                 | Modelscope                                                                                      | HuggingFace                                                                                |
|---------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| bird train and dev dataset            | [🤖 Modelscope](https://modelscope.cn/datasets/cycloneboy/bird_train)                           | [🤗 HuggingFace](https://huggingface.co/datasets/cycloneboy/bird_train) |
| CscSQL-Merge-Qwen2.5-Coder-3B-Instruct | [🤖 Modelscope](https://modelscope.cn/models/cycloneboy/CscSQL-Merge-Qwen2.5-Coder-3B-Instruct) | [🤗 HuggingFace](https://huggingface.co/cycloneboy/CscSQL-Merge-Qwen2.5-Coder-3B-Instruct) |
| CscSQL-Merge-Qwen2.5-Coder-7B-Instruct | [🤖 Modelscope](https://modelscope.cn/models/cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct) | [🤗 HuggingFace](https://huggingface.co/cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct) |
| CscSQL-Grpo-Qwen2.5-Coder-3B-Instruct | [🤖 Modelscope](https://modelscope.cn/models/cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-3B-Instruct)  | [🤗 HuggingFace](https://huggingface.co/cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-3B-Instruct)  |
| CscSQL-Grpo-XiYanSQL-QwenCoder-3B-2502 | [🤖 Modelscope](https://modelscope.cn/models/cycloneboy/CscSQL-Grpo-XiYanSQL-QwenCoder-3B-2502) | [🤗 HuggingFace](https://huggingface.co/cycloneboy/CscSQL-Grpo-XiYanSQL-QwenCoder-3B-2502) |
| CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct | [🤖 Modelscope](https://modelscope.cn/models/cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct)  | [🤗 HuggingFace](https://huggingface.co/cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct)  |
| CscSQL-Grpo-XiYanSQL-QwenCoder-7B-2502 | [🤖 Modelscope](https://modelscope.cn/models/cycloneboy/CscSQL-Grpo-XiYanSQL-QwenCoder-7B-2502) | [🤗 HuggingFace](https://huggingface.co/cycloneboy/CscSQL-Grpo-XiYanSQL-QwenCoder-7B-2502) |

## Train and Eval

### Eval docs

- [Reference data processing](docs/process_data.md)
- [Eval bird ](docs/eval_bird.md)

### Train docs

- [TODO]()

## TODO

- [x] Release inference code
- [x] Upload Model
- [ ] Fix bug
- [ ] Release training code

## Thanks to the following projects

- [open-r1](https://github.com/huggingface/open-r1)
- [OmniSQL](https://github.com/RUCKBReasoning/OmniSQL)

## Citation

```bibtex
@misc{sheng2025cscsqlcorrectiveselfconsistencytexttosql,
      title={CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning}, 
      author={Lei Sheng and Shuai-Shuai Xu},
      year={2025},
      eprint={2505.13271},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.13271}, 
}
```