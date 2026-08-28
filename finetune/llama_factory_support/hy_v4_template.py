"""
HYV4 chat template registration for LLaMA Factory.

Usage:
    1. Copy this file's register_template block into LLaMA Factory's
       src/llamafactory/data/template.py  (for upstream MR).
    2. Or import this module before training to register at runtime:
       import hy_v4_template
"""

from llamafactory.data.template import ReasoningTemplate, register_template
from llamafactory.data.formatter import EmptyFormatter, StringFormatter


# ---------------------------------------------------------------------------
# HYV4 (MoE, pure text) chat template
#
# Token format (from chat_template.jinja & tokenizer_config.json):
#   Each turn:  <｜hy_start:opensource｜>{role}<｜hy_middle:opensource｜>{content}<｜hy_end:opensource｜>
#   BOS:        <｜hy_start:opensource｜>  (token ID 120000)
#   Middle:     <｜hy_middle:opensource｜> (token ID 120001)
#   EOS:        <｜hy_end:opensource｜>    (token ID 120025)
#
# Loss mask: only compute loss on assistant content (including eos).
#
# Reasoning / Slow-thinking support:
#   This template uses ReasoningTemplate with thought_words so that LLaMA
#   Factory can correctly mask think-tag tokens during loss computation.
#   - thought_words: ("<think:opensource>", "</think:opensource>")
#   - enable_thinking: set globally via data_args.enable_thinking (default True)
#
#   IMPORTANT: To train slow-thinking (chain-of-thought) behaviour, your
#   training data must include <think:opensource>...</think:opensource> tags
#   inside the assistant content.  If your data does NOT contain think tags,
#   the model will only learn fast-thinking (direct answer) mode.
#   The `reasoning_effort` field from the API is NOT used by LLaMA Factory;
#   slow-vs-fast is determined solely by the presence of think tags in data.
# ---------------------------------------------------------------------------

register_template(
    name="hy_v4",
    template_class=ReasoningTemplate,
    format_user=StringFormatter(slots=["<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{{content}}<｜hy_end:opensource｜>"]),
    format_assistant=StringFormatter(slots=["<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>{{content}}<｜hy_end:opensource｜>"]),
    format_system=StringFormatter(slots=["<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>{{content}}<｜hy_end:opensource｜>"]),
    format_prefix=EmptyFormatter(slots=[]),
    thought_words=("<think:opensource>", "</think:opensource>"),
    stop_words=["<｜hy_end:opensource｜>"],
    efficient_eos=False,
)
