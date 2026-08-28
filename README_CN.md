<p align="left">
   <a href="README.md">English</a>&nbsp;｜&nbsp;中文
</p>
<br>

<p align="center">
 <img src="assets/logo-zh.png" width="400"/> <br>
</p>

<div align="center" style="line-height: 1;">


[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#许可证)
&nbsp;&nbsp;
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Tencent%20Hy-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/tencent/Hy4-preview)
&nbsp;&nbsp;
[![ModelScope](https://img.shields.io/badge/ModelScope-Tencent%20Hy-624aff)](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview)
&nbsp;&nbsp;
[![cnb.cool](https://img.shields.io/badge/cnb.cool-Tencent%20Hy-blue?logoColor=white)](https://cnb.cool/ai-models/tencent/Hy4-preview)
&nbsp;&nbsp;
[![GitCode](https://img.shields.io/badge/GitCode-Tencent%20Hy-red?logoColor=white)](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview)

</div>

<p align="center">
    🖥️&nbsp;<a href="https://aistudio.tencent.com/"><b>官方网站</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    💬&nbsp;<a href="https://github.com/Tencent-Hunyuan/Hy4-preview"><b>GitHub</b></a></p>

---

## 目录

- [模型介绍](#模型介绍)
- [新一代旗舰模型](#新一代旗舰模型)
- [为生产力而生](#为生产力而生)
- [Benchmark 附录](#benchmark-附录)
- [已知局限](#已知局限)
- [新闻](#新闻)
- [模型链接](#模型链接)
- [快速开始](#快速开始)
- [推理和部署](#推理和部署)
  - [vLLM](#vllm)
  - [SGLang](#sglang)
- [模型微调](#模型微调)
- [量化工具](#量化工具)
- [许可证](#许可证)
- [联系我们](#联系我们)

---

## 模型介绍

**Hy4 preview** 是由腾讯混元团队研发的新一代混合专家（MoE）旗舰模型。模型总参数量 770B，每个 token 激活 49B，主干共包含78层，第一层采用标准 FFN，其余 77 层均为 MoE 结构，每层包含 256 个路由专家与 1 个共享专家，每个 token 激活 top-8 路由专家及共享专家。主干之外原生内置 1 层 MTP（总参数量 10B，激活 0.7B）以支持投机解码。

在架构层面，受到 DeepSeek 和 GLM 的启发，注意力侧采用 Gated [DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556)（Gated DSA），并引入 [IndexCache](https://arxiv.org/abs/2603.12201) 跨层复用稀疏索引，残差侧采用 [iHC (identity Hyper-Connections)](https://zhuanlan.zhihu.com/p/2010852389670908320) 扩展层间信息通路。

### 模型规格

> 下表仅列出主干网络参数，不含MTP层

| 属性 | 值 |
|:---|:---|
| 架构 | 混合专家（MoE） |
| 总参数 | 770B |
| 激活参数量 | 49B |
| 层数 | 78 |
| 隐藏层维度 | 6144 |
| 注意力类型 | Gated DSA |
| 注意力头数 | 64 |
| Query 压缩维度 | 2048 |
| Key-Value 压缩维度 | 512 |
| Indexer 头数 / 头维度 | 32 / 128 |
| Indexer top-k | 2048 |
| 残差流数 | 4 |
| 路由专家数 | 256 |
| 共享专家数 | 1 |
| 单 token 激活路由专家数 | 8 |
| MoE中间层维度 | 2048 |
| FFN中间层维度 | 18432 |
| 上下文长度 | 1M |
| 词表大小 | 120832 |

## 新一代旗舰模型

Hy4 preview 在模型尺寸、上下文长度、数据规模上都进行了显著的扩展，预训练和后训练的共同进步带来了智能水平的又一次巨大提升，稳居开源模型第一梯队。

<p align="center">
  <img src="assets/benchmark.jpg" width="100%"/>
</p>

## 为生产力而生

通过与腾讯内部软件工程师、游戏开发者、金融分析师、安全专家等各领域顶尖专家的高质量数据共建，Hy4 preview 在各类真实生产力任务上取得显著进步：

**软件工程**：增强长程开发任务的理解、规划、调试与验证能力，进一步提升前端开发的视觉审美和交互质量。

**办公分析**：显著提升复杂办公环境理解和金融分析能力，着重优化数据分析、跨文件协作，完成从信息处理到文档、表格与演示文稿交付的完整流程。

**游戏开发**：增强一句需求直接生成可玩原型的能力，并能熟练使用游戏引擎，开发者可以通过多轮交互持续完善复杂游戏项目。

**科学研究**：显著提升复杂科研问题的理解、推理与求解能力，模型在 AI 研发、分子动力学模拟、凝聚态物理、基础数学等各类场景中均有长足进步。

同时，Hy4 preview 持续与 CodeBuddy / WorkBuddy 等产品深度协同，优化生产力场景的真实用户体验。为验证这一点，我们组织了 163 位内部专家基于 203 个真实工程任务进行模型盲测：Hy4 preview（均分 2.99 / 4）略优于 GLM 5.3（均分 2.92 / 4；胜 46.8% / 平 12.8% / 负 40.4%）和 Kimi K3（均分 2.94 / 4；胜 51.2% / 平 7.9% / 负 40.9%）。

## Benchmark 附录

<p align="center">
  <img src="assets/benchmark-appendix.jpg" width="100%"/>
</p>

## 已知局限

Hy4 preview 是 Hy4 迭代的一个早期版本，预训练和后训练均仍有较大的提升空间，也有一些已知问题，如复杂任务的长思考和过度自我验证倾向，我们将持续敏捷迭代。如同 Hy3 preview，我们希望通过 Hy4 preview 的尽快发布获得广泛的真实反馈，从而显著提升 Hy4 正式版。同时，我们将坚持发挥与腾讯产品和专家深度合作的独特优势，持续提升生产力的普惠性和上限。

## 新闻

* 🔥 我们在 [Hugging Face](https://huggingface.co/tencent/Hy4-preview)、[ModelScope](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview)、[GitCode](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview) 和 [CNB](https://cnb.cool/ai-models/tencent/Hy4-preview) 开源了 **Hy4 preview** 和 **Hy4 preview-FP8** 模型权重。

## 模型链接

| 模型名 | 简介 | Hugging Face | ModelScope | GitCode | CNB |
|:---|:---|:---:|:---:|:---:|:---:|
| Hy4 preview | Instruct 模型 | 🤗 [Model](https://huggingface.co/tencent/Hy4-preview) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview) | [Model](https://cnb.cool/ai-models/tencent/Hy4-preview) |
| Hy4 preview-FP8 | FP8 量化 Instruct 模型 | 🤗 [Model](https://huggingface.co/tencent/Hy4-preview-FP8) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview-FP8) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview-FP8) | [Model](https://cnb.cool/ai-models/tencent/Hy4-preview-FP8) |

## 快速开始

建议先通过 [vLLM](#vllm) 或 [SGLang](#sglang) 部署服务，然后通过 OpenAI 兼容 API 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="hy4-preview",
    messages=[
        {"role": "user", "content": "你好！请简单介绍一下你自己。"},
    ],
    temperature=0.9,
    top_p=1.0,
)
print(response.choices[0].message.content)
```

> **推荐参数**：`temperature=0.9`，`top_p=1.0`。
>
> **推理模式**：默认为 `"high"`（深度思维链），适合数学、编程、推理等复杂任务；日常对话如需直接回复，可传入 `extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}}`。

具体部署方式请参考下方[推理和部署](#推理和部署)章节。

## 推理和部署

对于生产环境部署，我们建议使用 [vLLM](https://github.com/vllm-project/vllm) 或 [SGLang](https://docs.sglang.io/)。请参考部署方案：
- [Hy4-Preview vLLM Recipe](https://recipes.vllm.ai/tencent/Hy4-preview)
- [Hy4-Preview SGLang Cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Tencent/Hy4-Preview)


### vLLM

使用社区官方镜像部署：`vllm/vllm-openai:hy4-preview`：

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:hy4-preview tencent/Hy4-preview-FP8 \
    --tensor-parallel-size 8 \
    --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}' \
    --attention-backend FLASHMLA_SPARSE \
    --tool-call-parser hy_v4 \
    --reasoning-parser hy_v4 \
    --enable-auto-tool-choice \
    --port 8000 \
    --served-model-name hy4-preview
```


### SGLang

使用社区官方镜像部署：`lmsysorg/sglang:hy4-preview`，（支持 x86 和 Arm）。

```bash
docker pull lmsysorg/sglang:hy4-preview

docker run --gpus all --ipc=host -p 8000:8000 lmsysorg/sglang:hy4-preview \
  python3 -m sglang.launch_server \
    --model tencent/Hy4-preview-FP8 \
    --tp-size 8 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --port 8000 \
    --served-model-name hy4-preview
```

## 模型微调

Hy4 preview 提供了完整的模型微调流程，详细的微调文档请参考：[模型微调指南](./finetune/README_CN.md)

## 量化工具

我们提供了 [AngelSlim](https://github.com/tencent/AngelSlim)——一套易用、全面、高效的大模型压缩工具包，涵盖常用量化算法、低比特量化和投机采样等能力。

## 许可证

Hy4 preview 基于 **Apache 2.0 许可证** 发布。详情请参阅 [LICENSE](./LICENSE)。

## 联系我们

如有问题或建议，欢迎通过邮件联系我们的研发和产品团队：

📧 **hunyuan_opensource@tencent.com**

---

<p align="center">
  <i>Hy4 preview 由腾讯混元团队研发。</i>
</p>
