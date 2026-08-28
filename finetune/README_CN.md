<p align="left">
   <a href="README.md">English</a> ｜ 中文
</p>

# 模型微调

Hy4 preview 提供了模型微调相关流程，您可以在此章节对训练数据格式进行处理以供模型微调使用。

## 训练数据格式及处理

**Hy4 preview 同时支持慢思考与快思考两种模式，可通过 `reasoning_effort` 参数控制（可选值：`high`、`no_think`）。**

训练数据按照以下形式处理为 messages 格式，训练和推理的默认 system prompt 为空，可以根据自己的需求进行设定。

```python
# Fast thinking pattern (no_think)
{"reasoning_effort": "no_think", "messages": [{"content": "你是一个有用的人工智能助手。\n现在的时间是2026-01-01 13:26:12 周四", "role": "system"}, {"content": "1+1=?", "role": "user"}, {"role": "assistant", "content": "1+1=2"}]}

# Slow thinking pattern (high)
{"reasoning_effort": "high", "messages": [{"content": "你是一个有用的人工智能助手。\n现在的时间是2026-01-01 13:26:12 周四", "role": "system"}, {"content": "1+1=?", "role": "user"}, {"role": "assistant", "content": "1+1=2", "reasoning_content": "用户问的是1+1等于多少。在基本的十进制算术中，1+1等于2。"}]}
```

使用 `apply_chat_template` 进行 tokenize 的示例：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./models", use_fast=False, trust_remote_code=True)

messages = [
    {"content": "你是一个有用的人工智能助手。", "role": "system"},
    {"content": "1+1=?", "role": "user"},
    {"role": "assistant", "content": "1+1=2"}
]
ids = tokenizer.apply_chat_template(messages, tokenize=True, reasoning_effort="no_think")
```

## 微调流程

### 硬件需求

经过测试，最小的资源配置如下：

- **LoRA 微调**：最少需要 8 机 64 卡，每卡显存至少 96GB，每机 CPU 内存至少 2TB。
- **全量微调**：最少需要 16 机 128 卡，每卡显存至少 96GB，每机 CPU 内存至少 2TB。

> 注：以上为最小资源配置，实际所需资源会随 max_seq_length、batch size 等因素相应增加。

### 配置机器间免密 ssh 登录（多机训练）

> 如果只使用单机训练，可跳过本节。

以下操作以两个机器为例，两台机器的 ip 分别以`${ip1}`和`${ip2}`标识，以下操作均在 docker container 内执行。

首先，配置多机container免密，在每台机器上执行。

```sh
ssh-keygen			# 生成id_rsa和id_rsa.pub，用于免密登录
ssh-keygen -t rsa -A    # 生成/etc/ssh/ssh_host_rsa_key和ssh_host_ecdsa_key， 用于后面启动ssh listen
/usr/sbin/sshd -p 36005 -o ListenAddress=0.0.0.0        # 启动 SSH 监听
echo "Port 36005" > ~/.ssh/config   # ssh 连接端口修改为 36005
passwd root    # 需要配置root密码，否则监测平台会报警
```

注意：这里的`36005`是一个示例端口，可以选用任意端口，但需要保证使用的端口**开放**且**不被其他的进程占用**。

接下来，在每台机器的 container 内，执行：

```sh
cat ~/.ssh/id_rsa.pub
```

**将输出的 ssh 公钥复制并粘贴到`~/.ssh/authorized_keys`文件中，每行一个公钥，每台机器上都要做这个操作**。最终每台机器上的`~/.ssh/authorized_keys`文件内容应当是一致的，并且包含了所有机器的公钥。

需要注意，多节点训练时，每个节点上执行的代码都得一致，建议挂载一个共享的网络盘，如果无法挂载共享网盘，则需要手动将数据集、脚本、代码复制在多台机器的相同目录下。

### 启动方式

本项目提供三种微调方式，您可以根据需求选择：

- **DeepSpeed 原生微调**（基于 HuggingFace Transformers Trainer）：位于 `deepspeed_support` 目录下
- **LLaMA-Factory 微调**：位于 `llama_factory_support` 目录下
- **ms-swift 微调**：位于 `ms_swift_support` 目录下

#### DeepSpeed 原生微调

参考：[HuggingFace Transformers Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)

##### 单机启动微调

在 `deepspeed_support` 目录下，执行：

```sh
pip install -r requirements.txt
bash train.sh
```

##### 多机启动微调

如果要用多台机器启动微调，请先完成 [配置机器间免密 ssh 登录](#配置机器间免密-ssh-登录多机训练) 中的配置，并保证多台机器在一个集群内。

确认依赖已经安装完成（如未安装，请执行`pip install -r requirements.txt`安装），然后在`train.sh`中的开头增加以下配置：

```shell
export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "192.168.1.1,192.168.1.2" or single node "192.168.1.1"
IP_LIST=${IP_LIST:-"127.0.0.1"}
```

注意：如果`IP_LIST`环境变量未设置，则将`IP_LIST`替换为IP列表！格式为：
```
如果只有一个IP：
IP_LIST=${ip_1}

如果有多个IP：
IP_LIST=${ip_1},${ip_2}

```

请将`${ip_1}`和`${ip_2}`替换为真实的IP地址。

然后，在`${ip1}`的机器上，在`deepspeed_support/`目录下，执行`bash train.sh`即可，注意第一次启动时可能会看见以下的输出：

```ssh
The authenticity of host '[ip]:36005 ([ip]:36005)' can't be established.
ECDSA key fingerprint is xxxxxx.
ECDSA key fingerprint is MD5:xxxxxx.
Are you sure you want to continue connecting (yes/no)?
```

此时输入`yes`即可继续。

##### 关键参数

脚本中的关键参数如下：

- `--deepspeed`: 此参数应当指向一个 deepspeed 的配置文件，`deepspeed_support`文件夹下提供了四种 DeepSpeed 的默认配置文件：`ds_zero2_no_offload.json`, `ds_zero2_offload.json`, `ds_zero3_no_offload.json`, `ds_zero3_offload.json`，可根据显存与通信情况选择不同的 ZeRO 阶段（ZeRO-2 / ZeRO-3）与 offload 策略
- `--model_name_or_path`: 要加载的 Hy4 preview 的 HF 预训练模型权重，否则无法加载
- `--tokenizer_name_or_path`: tokenizer 文件夹路径, 否则无法加载
- `--train_data_file`: 训练文件路径，应该为一个 jsonl 文件
- `--output_dir`: 输出文件夹，log、tensorboard 和权重都会存储在这个路径下
- `--per_device_train_batch_size`: 每张卡上的 batch size
- `--gradient_accumulation_steps`: 梯度累计次数，`per_device_train_batch_size * gradient_accumulation_steps * dp_size`为 global_batch_size
- `--max_steps`: 训练的总步数
- `--save_steps`: 每多少个 step 存储一个 checkpoint
- `--use_lora`: 是否用 lora 训练，同时接收`--lora_rank`，`--lora_alpha`和`--lora_dropout`参数。lora 默认应用于 MLA（Multi-head Latent Attention）投影层："q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"，如果需要改变的话在代码中修改即可。注意：**使用 lora 训练时，只会保存 lora 的权重，而不会保存 base 模型的权重**
- `--make_moe_param_leaf_module`：当用 zero3 以及 MoE 训练时，将 MoE 模块视作一个 leaf module，即它的参数不进行 zero3 切分，这个选项预计会显著增加显存占用
- `--gradient_checkpointing`：开启梯度检查点- `--learning_rate`: 训练时的最大学习率
- `--min_lr`: 训练时的最小学习率
- `--use_flash_attn`: 开启 flash-attention 进行训练加速

**注意：**

- 如果想从一个中途保存的 ckpt 继续训练，而不是加载一个预训练的权重，直接指定`--resume_from_checkpoint`为之前训练保存的 ckpt 路径，不要指定`--model_name_or_path`，这样只会加载权重，而不会加载训练状态
- 从 ckpt 继续训练时，loss 可能会有微小的偏差，这是由一些非确定性算法带来的随机性，是正常现象。参考：[HuggingFace Transformers Trainer Randomness](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#randomness)
- 当 `--model_name_or_path` 有效时，所有模型相关的参数都会被忽略
- 一个 batch 内的样本会通过 padding 对齐 batch 内最长的样本，而每条样本的长度最长为 max_seq_length，超出的部分会被裁剪
- 如果报出**线性层** bias 权重没有 load 的 warning，忽略即可，Hy4 preview 的线性层（q_a_proj / q_b_proj / kv_a_proj_with_mqa / kv_b_proj / o_proj 等）不使用 bias。注意：MoE 路由的 `e_score_correction_bias` 属于 buffer，已由训练脚本自动加载，如果加载失败请不要忽略。

##### 显存不足怎么办？

参考：[DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)

可以尝试修改 ds config，去掉这几个参数的 auto 属性，改小试试看：

- `stage3_param_persistence_threshold`
- `stage3_prefetch_bucket_size`
- `stage3_max_reuse_distance`

#### LLaMA-Factory 微调

如果对 LLaMA-Factory 较为熟悉，可使用 LLaMA-Factory 进行微调。脚本、代码以及配置文件都归档在 `llama_factory_support` 目录下。如果没有特别说明，接下来我们提到的文件都是该目录下的文件。

##### 安装

可以通过下载源码 https://github.com/hiyouga/LLaMA-Factory/tree/main ，根据网站的指引进行安装。

##### 配置文件

我们提供了 llama-factory 的微调示例配置文件 `hy_v4_lora_sft.yaml`和`hy_v4_full_sft.yaml`文件，分别对应 LoRA 微调和全量微调。

脚本中的关键参数如下：

**模型相关：**

- `model_name_or_path`: Hy4 preview HF 格式预训练模型权重路径
- `trust_remote_code`: 是否信任远程代码, Hy4 preview 需要设置为 `true`

**训练方法：**

- `stage`: 训练阶段, 当前为 `sft`(监督微调)
- `finetuning_type`: 微调类型, 可选 `full`(全量微调) 或 `lora`(LoRA 微调)
- `deepspeed`: DeepSpeed 配置文件路径, 全量微调推荐 `../deepspeed_support/ds_zero3_offload.json`
- `fsdp` + `fsdp_config`: FSDP 分布式策略, LoRA 微调推荐使用 FSDP（配置已内置在 `hy_v4_lora_sft.yaml` 中）; 与 DeepSpeed 二选一

> **分布式策略选择建议：**
> - **FSDP**：推荐用于 LoRA 微调，兼容性好，配置简单
> - **DeepSpeed ZeRO-3 + Offload**：推荐用于全量微调或显存紧张的场景

**LoRA 参数(仅 LoRA 微调时生效)：**

- `lora_rank`: LoRA 秩, 默认 `64`
- `lora_alpha`: LoRA alpha 系数, 默认 `128`
- `lora_dropout`: LoRA dropout 比率, 默认 `0.05`
- `lora_target`: LoRA 应用的目标模块, 默认为 `q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj`

**数据集：**

- `dataset_dir`: 数据集目录路径
- `dataset`: 数据集名称, 需要在 `dataset_dir` 下的 `dataset_info.json` 中注册
- `template`: 对话模板, Hy4 preview 使用 `hy_v4`
- `cutoff_len`: 最大序列长度, 超出部分会被截断; LoRA 微调建议适当减小以节省显存
- `max_samples`: 每个数据集最多使用的样本数
- `overwrite_cache`: 是否覆盖已缓存的预处理数据集

**输出：**

- `output_dir`: 输出目录, 日志、TensorBoard 和权重都会存储在此路径下
- `logging_steps`: 每多少步记录一次日志
- `save_steps`: 每多少步保存一次 checkpoint
- `plot_loss`: 是否绘制训练 loss 曲线
- `overwrite_output_dir`: 是否覆盖已有的输出目录
- `save_only_model`: 是否只保存模型权重(不保存优化器状态等)
- `report_to`: 日志上报工具, 可选 `none`, `wandb`, `tensorboard`, `swanlab`, `mlflow`

**训练超参数：**

- `per_device_train_batch_size`: 每张卡上的 batch size
- `gradient_accumulation_steps`: 梯度累积步数, `per_device_train_batch_size * gradient_accumulation_steps * dp_size` 为 global batch size
- `learning_rate`: 最大学习率, 全量微调推荐 `1.0e-5`, LoRA 微调推荐 `2.0e-4`
- `num_train_epochs`: 训练轮数
- `lr_scheduler_type`: 学习率调度器类型, 推荐使用 `cosine_with_min_lr`
- `lr_scheduler_kwargs.min_lr_rate`: 最小学习率与最大学习率的比值, 例如 `0.1` 表示最小学习率为最大学习率的 10%
- `warmup_steps`: 预热步数
- `bf16`: 是否使用 BFloat16 混合精度训练
- `gradient_checkpointing`: 是否开启梯度重计算以节省显存
- `ddp_timeout`: 分布式训练超时时间(毫秒)
- `flash_attn`: 注意力实现方式, 推荐 `auto`(自动选择) 或 `sdpa`
- `resume_from_checkpoint`: 从指定 checkpoint 路径恢复训练, 设为 `null` 表示从头开始训练

##### 启动微调

如需多机训练，请先完成 [配置机器间免密 ssh 登录](#配置机器间免密-ssh-登录多机训练) 中的配置（单机训练可跳过此步骤）。

修改`train_lf.sh`中开头的以下配置：

```shell
export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "192.168.1.1,192.168.1.2" or single node "192.168.1.1"
export IP_LIST=${IP_LIST:-"127.0.0.1"}
```

注意：
1. 如果`IP_LIST`环境变量未设置，则将`IP_LIST`替换为IP列表！格式为：
```
如果只有一个IP：
IP_LIST=${ip_1}

如果有多个IP：
IP_LIST=${ip_1},${ip_2}

```
请将`${ip_1}`和`${ip_2}`替换为真实的IP地址。

2. 如需指定微调配置文件，可通过`YAML_FILE`环境变量设置，默认为`hy_v4_full_sft.yaml`。例如使用 LoRA 微调配置：
```shell
export YAML_FILE=hy_v4_lora_sft.yaml
```

然后，在每一台机器上，在`llama_factory_support/`目录下执行启动脚本：

```shell
bash train_lf.sh
```

#### ms-swift 微调

如果对 ms-swift 较为熟悉，可使用 ms-swift 进行微调。脚本、代码以及配置文件都归档在 `ms_swift_support` 目录下。如果没有特别说明，接下来我们提到的文件都是该目录下的文件。

##### 安装

可以通过 pip 安装 ms-swift：

```sh
pip install ms-swift
```

或从源码安装：https://github.com/modelscope/ms-swift

##### 微调脚本与配置文件

| 微调方式 | 配置文件 | 启动脚本 |
|---------|---------|---------|
| 全量微调 | `hy_v4_full_sft.yaml` | `bash sft_train.sh` |
| LoRA 微调 | `hy_v4_lora_sft.yaml` | `bash sft_train_lora.sh` |

##### 关于 eos_token_id Patch

目录下的 `hy_v4_swift_patches.py` 文件用于修复 ms-swift 默认模板中 eos token 的问题。默认模板将 `<｜hy_eos｜>` 字符串作为 `chat_sep` 和 `suffix`，该字符串会被 tokenize 为多个 token ID，导致推理时 `model.generate()` 无法正确停止。

Patch 通过 `[['eos_token_id']]` 语法重新注册模板，使 ms-swift 在运行时动态解析 `tokenizer.eos_token_id`，生成正确的单个 token。

启动脚本已通过 `--custom_register_path hy_v4_swift_patches.py` 自动加载此 patch，无需额外操作。

##### 关键参数

配置文件中的关键参数如下：

**模型相关：**

- `model`: 模型路径，可以是 HuggingFace Hub ID 或本地路径
- `model_type`: 模型类型，设为 `hy_v4`
- `template`: 对话模板，设为 `hy_v4`
- `torch_dtype`: 数据类型，推荐 `bfloat16`
- `attn_impl`: 注意力实现，推荐 `sdpa`

**训练方法：**

- `train_type`: 微调类型，全量微调设为 `full`，LoRA 微调设为 `lora`
- `lora_rank`: LoRA 秩，默认 `64`
- `lora_alpha`: LoRA alpha 系数，默认 `128`
- `lora_dropout`: LoRA dropout 比率，默认 `0.05`

**数据集：**

- `dataset`: 数据集路径，支持本地 jsonl 文件（sharegpt 格式）
- `max_length`: 最大序列长度，超出部分会被截断
- `lazy_tokenize`: 是否延迟 tokenize，推荐 `true`

**输出：**

- `output_dir`: 输出目录
- `save_steps`: 每多少步保存一次 checkpoint
- `save_total_limit`: 最多保留的 checkpoint 数量
- `logging_steps`: 每多少步记录一次日志
- `report_to`: 日志上报工具，可选 `none`, `wandb`, `tensorboard`, `swanlab`, `mlflow`

**训练超参数：**

- `per_device_train_batch_size`: 每张卡上的 batch size
- `gradient_accumulation_steps`: 梯度累积步数
- `learning_rate`: 最大学习率，全量微调推荐 `1.0e-5`，LoRA 微调推荐 `2.0e-4`
- `num_train_epochs`: 训练轮数
- `lr_scheduler_type`: 学习率调度器类型，推荐 `cosine`
- `warmup_steps`: 预热步数
- `bf16`: 是否使用 BFloat16 混合精度训练

**分布式策略 / 优化：**

- `deepspeed`: DeepSpeed 策略，可选 `zero0`, `zero2`, `zero2_offload`, `zero3`, `zero3_offload`；全量微调推荐 `zero3_offload`
- `fsdp` + `fsdp_config`: FSDP 分布式策略，LoRA 微调推荐使用 FSDP；与 DeepSpeed 二选一
- `gradient_checkpointing`: 是否开启梯度重计算
- `max_grad_norm`: 梯度裁剪阈值

> **分布式策略选择建议：**
> - **FSDP**：推荐用于 LoRA 微调，兼容性好，配置简单
> - **DeepSpeed ZeRO-3 + Offload**：推荐用于全量微调或显存紧张的场景

**其他：**

- `ddp_timeout`: 分布式训练超时时间（毫秒）
- `seed`: 随机种子
- `resume_from_checkpoint`: 从指定 checkpoint 路径恢复训练

##### 启动微调

如需多机训练，请先完成 [配置机器间免密 ssh 登录](#配置机器间免密-ssh-登录多机训练) 中的配置（单机训练可跳过此步骤）。

修改 `sft_train.sh` 脚本中的以下配置：

```shell
export HOST_GPU_NUM=8
# IP list, comma separated. e.g. "10.0.0.1,10.0.0.2" or single node "127.0.0.1"
export IP_LIST=${IP_LIST:-"127.0.0.1"}
```

然后，在每一台机器上，在 `ms_swift_support/` 目录下执行启动脚本：

```sh
# 单机训练
bash sft_train.sh

# 多机训练（在每台机器上执行）
IP_LIST="10.0.0.1,10.0.0.2" bash sft_train.sh
```