<p align="left">
    <a href="README_CN.md">中文</a>&nbsp;｜&nbsp;English
</p>
<br>

<p align="center">
 <img src="assets/logo-en.png" width="400"/> <br>
</p>

<div align="center" style="line-height: 1;">


[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#license)
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
    🖥️&nbsp;<a href="https://aistudio.tencent.com/"><b>Official Website</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    💬&nbsp;<a href="https://github.com/Tencent-Hunyuan/Hy4-preview"><b>GitHub</b></a></p>

---

## Table of Contents

- [Model Introduction](#model-introduction)
- [A New Flagship Generation](#a-new-flagship-generation)
- [Built for Productivity](#built-for-productivity)
- [Benchmark Appendix](#benchmark-appendix)
- [Known Limitations](#known-limitations)
- [News](#news)
- [Model Links](#model-links)
- [Quickstart](#quickstart)
- [Deployment](#deployment)
  - [vLLM](#vllm)
  - [SGLang](#sglang)
- [Finetuning](#finetuning)
- [Quantization](#quantization)
- [License](#license)
- [Contact Us](#contact-us)

---

## Model Introduction

**Hy4 preview** is a new-generation Mixture-of-Experts (MoE) flagship model developed by the Tencent Hy Team. The model comprises 770B total parameters, of which 49B are activated per token. The backbone consists of 78 layers, where the first layer uses a standard dense FFN and the remaining 77 layers replace it with MoE, each containing 256 routed experts and 1 shared expert; every token activates the top-8 routed experts along with the shared expert. In addition to the backbone, 1 native MTP layer (10B total parameters, 0.7B activated) is built in for speculative decoding.

On the architecture side, inspired by DeepSeek and GLM, the attention module employs Gated [DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556) (Gated DSA) with [IndexCache](https://arxiv.org/abs/2603.12201) for cross-layer sparse index reuse. The residual pathway uses [iHC (identity Hyper-Connections)](https://zhuanlan.zhihu.com/p/2010852389670908320) to expand inter-layer information flow.

### Model Specifications

> The table below lists backbone parameters only, excluding the MTP layer.

| Property | Value |
|:---|:---|
| Architecture | Mixture-of-Experts (MoE) |
| Total Parameters | 770B |
| Activated Parameters | 49B |
| Layers | 78 |
| Hidden Size | 6144 |
| Attention Type | Gated DSA |
| Attention Heads | 64 |
| Query Compression Dimension | 2048 |
| Key-Value Compression Dimension | 512 |
| Indexer Heads / Head Dimension | 32 / 128 |
| Indexer top-k | 2048 |
| Residual Streams | 4 |
| Routed Experts | 256 |
| Shared Experts | 1 |
| Activated Routed Experts per Token | 8 |
| MoE Intermediate Size | 2048 |
| FFN Intermediate Size | 18432 |
| Context Length | 1M |
| Vocabulary Size | 120832 |

## A New Flagship Generation

We scaled Hy4 preview on three fronts: model size, context length, and training data. Stronger pre-training and a substantially larger post-training run compound into another step change in capability — the largest generation-over-generation gain we've measured, and enough to put Hy4 preview at the open-source frontier.

<p align="center">
  <img src="assets/benchmark.jpg" width="100%"/>
</p>

## Built for Productivity

We partnered with top experts inside Tencent — such as software engineers, game developers, finance analysts, and security experts — and built training data around the work they ship. The result is a model that gets meaningfully further on the tasks these teams run every day:

**Software engineering**: Better at understanding, planning, debugging, and verifying long-horizon development tasks, with further gains in the visual taste and interaction quality of front-end work.

**Office and analysis**: Takes messy context spread across many files and converts it into shareable artifacts — documents, spreadsheets, and presentations — handling data analysis, equations, and financial models with greater precision.

**Game development**: Turns a single prompt into a playable prototype and works fluently with game engines, so developers can keep refining complex projects over multiple turns.

**Scientific research**: Stronger understanding, reasoning, and problem-solving on hard research questions, with solid progress across AI research, molecular dynamics, condensed matter physics, and pure mathematics.

We also continue to co-design Hy4 preview with Tencent products like CodeBuddy and WorkBuddy, so that gains in the model show up in the work people actually do with it. To check that, we ran a blind side-by-side evaluation: 163 internal experts rated model outputs on 203 engineering tasks. Hy4 preview came out slightly ahead of both GLM 5.3 (2.99 vs. 2.92 average, 46.8% wins / 12.8% ties / 40.4% losses) and Kimi K3 (2.99 vs. 2.94, 51.2% wins / 7.9% ties / 40.9% losses).

## Benchmark Appendix

<p align="center">
  <img src="assets/benchmark-appendix.jpg" width="100%"/>
</p>

## Known Limitations

This is an early version of Hy4. There is real headroom left in both pre-training and post-training, and we are shipping with known issues — among them, spending longer than necessary reasoning through complex tasks, and a tendency to over-verify its own work. We'll keep iterating quickly on these. As with Hy3 preview, we would rather ship early and hear what breaks — that's what made Hy3 substantially better, and it's how we will get Hy4 right. We will also keep collaborating closely with Tencent's products and in-house experts to push the boundaries of model intelligence while making it more abundant and affordable.

## News

* 🔥 We open-source **Hy4 preview** and **Hy4 preview-FP8** model weights on [Hugging Face](https://huggingface.co/tencent/Hy4-preview), [ModelScope](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview), [GitCode](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview), and [CNB](https://cnb.cool/ai-models/tencent/Hy4-preview).

## Model Links

| Model Name | Description | Hugging Face | ModelScope | GitCode | CNB |
|:---|:---|:---:|:---:|:---:|:---:|
| Hy4 preview | Instruct model | 🤗 [Model](https://huggingface.co/tencent/Hy4-preview) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview) | [Model](https://cnb.cool/ai-models/tencent/Hy4-preview) |
| Hy4 preview-FP8 | FP8 quantized instruct model | 🤗 [Model](https://huggingface.co/tencent/Hy4-preview-FP8) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview-FP8) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview-FP8) | [Model](https://cnb.cool/ai-models/tencent/Hy4-preview-FP8) |

## Quickstart

Deploy Hy4 preview with [vLLM](#vllm) or [SGLang](#sglang) first, then call the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="hy4-preview",
    messages=[
        {"role": "user", "content": "Hello! Can you briefly introduce yourself?"},
    ],
    temperature=0.9,
    top_p=1.0,
)
print(response.choices[0].message.content)
```

> **Recommended parameters**: `temperature=0.9`, `top_p=1.0`.
>
> **Reasoning mode**: Defaults to `"high"` (deep chain-of-thought), which suits complex tasks such as math, coding, and reasoning. For direct responses, pass `extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}}`.

See the [Deployment](#deployment) section below for how to start the API server.

## Deployment

For production serving, we recommend using [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://docs.sglang.io/). Please refer to the recipes:
- [Hy4-Preview vLLM Recipe](https://recipes.vllm.ai/tencent/Hy4-preview)
- [Hy4-Preview SGLang Cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Tencent/Hy4-Preview)

### vLLM

Build vLLM from source:
```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install --editable . --torch-backend=auto
```

Start the vLLM server with MTP enabled:

```bash
vllm serve tencent/Hy4-preview-FP8 \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --attention-backend FLASHMLA_SPARSE \
  --tool-call-parser hy_v4 \
  --reasoning-parser hy_v4 \
  --enable-auto-tool-choice \
  --port 8000 \
  --served-model-name hy4-preview
```

Or use official prebuilt image `vllm/vllm-openai:hy4-preview`:

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

Use the official prebuilt image `lmsysorg/sglang:hy4-preview` (multi-arch, x86 and Arm):

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

## Finetuning

Hy4 preview provides a complete model finetuning pipeline. For detailed documentation, please refer to: [Finetuning Guide](./finetune/README.md)

## Quantization

We provide [AngelSlim](https://github.com/tencent/AngelSlim), a more accessible, comprehensive, and efficient toolkit for large model compression. AngelSlim supports a comprehensive suite of compression tools for large-scale multimodal models, including common quantization algorithms, low-bit quantization, and speculative sampling.

## License

Hy4 preview is released under the **Apache License 2.0**. See [LICENSE](./LICENSE) for details.

## Contact Us

If you have any questions or suggestions, feel free to reach out to our R&D and product teams via email:

📧 **hunyuan_opensource@tencent.com**

---

<p align="center">
  <i>Hy4 preview is developed by the Tencent Hy Team.</i>
</p>
