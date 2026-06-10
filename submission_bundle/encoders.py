"""Encoder harness for the 3-family bundle — offline HF, protocol-pinned per family.

Every protocol knob lives in the family's runtime_meta.json ("encoder" block) so the
offline verify step can sweep configs against the TRAINING embedding caches and pin the
exact one (the binding constraint is matching the cache, not the model card):

  encoder_model_id      HF repo id (must be in models.txt)
  max_length            tokenizer truncation length
  pooling               "mean" | "last_token"
  pool_fp32             upcast hidden states to fp32 before pooling (official nemotron does)
  l2_normalize          normalize pooled vector (official nemotron/qwen3 do; trc5 does NOT)
  padding_side          "left" | "right" (qwen3 official: left)
  passage_prefix        prepended to the templated item text ("" for nemotron/lgai)
  batch_size            runtime encode batch
  bidirectional_llama   True only for nvidia/llama-embed-nemotron-8b (vendored trc5 fix)

The nemotron bidirectional block is vendored from the proven trc5 submission (model.py
lines ~1754-1945): transformers v5 removed `LlamaModel._update_causal_mask`, so we (a)
register a LlamaBidirectional config/model pair that forces `is_causal=False` everywhere
and overrides `_update_causal_mask` (v4 path), (b) monkey-patch the v5
`transformers.masking_utils` mask factories to emit a padding-only (non-causal) 4D mask,
and (c) defensively re-apply `is_causal=False` after `from_pretrained`.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import logging
from pathlib import Path

import numpy as np
import torch

LOG = logging.getLogger("encoders")


def resolve_cache_dir(bundle_dir) -> str | None:
    for cand in (_os.environ.get("HF_HOME"), "/app/hf_cache",
                 str(Path(bundle_dir) / ".hf_cache")):
        if not cand:
            continue
        try:
            Path(cand).mkdir(parents=True, exist_ok=True)
            return cand
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------------
# pooling (trc5 math; fp32/normalize variants are meta-gated)
# ---------------------------------------------------------------------------------
def _mean_pool(last_hidden, attention_mask, fp32: bool):
    if fp32:
        last_hidden = last_hidden.to(torch.float32)
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _last_token_pool(last_hidden, attention_mask, fp32: bool):
    if fp32:
        last_hidden = last_hidden.to(torch.float32)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    lens = (attention_mask.sum(dim=1) - 1).clamp(min=0)
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lens]


# ---------------------------------------------------------------------------------
# nemotron bidirectional fix (vendored from trc5 model.py — replicate EXACTLY)
# ---------------------------------------------------------------------------------
def _install_llama_bidirectional():
    from transformers import AutoConfig, AutoModel
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    def _nemo_bidir_mask_4d(attention_mask, dtype):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask
        m = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - m) * torch.finfo(dtype).min

    class LlamaBidirectionalConfig(LlamaConfig):
        model_type = "llama_bidirec"

        def __init__(self, pooling="avg", temperature=1.0, **kwargs):
            self.pooling = pooling
            self.temperature = temperature
            super().__init__(**kwargs)
            self.is_causal = False   # after super().__init__ (v5 power feature)

    class LlamaBidirectionalModel(LlamaModel):
        config_class = LlamaBidirectionalConfig

        def __init__(self, config):
            config.is_causal = False
            super().__init__(config)
            for layer in self.layers:
                layer.self_attn.is_causal = False

        def _update_causal_mask(self, attention_mask, input_tensor, cache_position=None,
                                past_key_values=None, output_attentions=False):
            return _nemo_bidir_mask_4d(attention_mask, input_tensor.dtype)

    def _nemo_bidir_create_causal_mask(config=None, input_embeds=None,
                                       attention_mask=None, *args, **kwargs):
        if input_embeds is None:
            input_embeds = kwargs.get("inputs_embeds")
        if input_embeds is None:
            return attention_mask
        return _nemo_bidir_mask_4d(attention_mask, input_embeds.dtype)

    try:
        import transformers.masking_utils as _mu
        for name in ("create_causal_mask", "create_sliding_window_causal_mask",
                     "create_chunked_causal_mask"):
            if hasattr(_mu, name):
                setattr(_mu, name, _nemo_bidir_create_causal_mask)
        LOG.info("nemotron: masking_utils factories patched (v5 path)")
    except ImportError:
        LOG.info("nemotron: no masking_utils (v4) — _update_causal_mask override active")

    for reg in ((AutoConfig.register, ("llama_bidirec", LlamaBidirectionalConfig)),
                (AutoModel.register, (LlamaBidirectionalConfig, LlamaBidirectionalModel))):
        try:
            reg[0](*reg[1])
        except (ValueError, KeyError):
            pass
    return LlamaBidirectionalModel


class Encoder:
    """One family's encoder: load once at import, embed batches of templated texts."""

    def __init__(self, enc_meta: dict, bundle_dir):
        from transformers import AutoModel, AutoTokenizer
        self.meta = dict(enc_meta)
        self.model_id = enc_meta["encoder_model_id"]
        self.max_length = int(enc_meta["max_length"])
        self.pooling = enc_meta.get("pooling", "mean")
        self.pool_fp32 = bool(enc_meta.get("pool_fp32", False))
        self.l2_normalize = bool(enc_meta.get("l2_normalize", False))
        self.batch_size = int(enc_meta.get("batch_size", 8))
        cache_dir = resolve_cache_dir(bundle_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (torch.bfloat16 if self.device == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=cache_dir, local_files_only=True)
        if enc_meta.get("padding_side"):
            self.tokenizer.padding_side = enc_meta["padding_side"]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if enc_meta.get("bidirectional_llama"):
            if self.model_id != "nvidia/llama-embed-nemotron-8b":
                raise RuntimeError("bidirectional_llama is nemotron-only")
            _install_llama_bidirectional()
            self.model = AutoModel.from_pretrained(
                self.model_id, cache_dir=cache_dir, torch_dtype=dtype,
                attn_implementation="sdpa", local_files_only=True)
            self.model.config.is_causal = False
            for layer in getattr(self.model, "layers", []):
                layer.self_attn.is_causal = False
        else:
            self.model = AutoModel.from_pretrained(
                self.model_id, cache_dir=cache_dir, torch_dtype=dtype,
                local_files_only=True)
        self.model.eval().to(self.device)
        LOG.info("encoder %s loaded (%s, pool=%s, max_len=%d)",
                 self.model_id, dtype, self.pooling, self.max_length)

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> np.ndarray:
        """[n, D] float32 embeddings (raw unless meta says l2_normalize)."""
        out = []
        for s in range(0, len(texts), self.batch_size):
            chunk = texts[s:s + self.batch_size]
            tok = self.tokenizer(chunk, padding="longest", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            hidden = self.model(**tok).last_hidden_state
            if self.pooling == "last_token":
                pooled = _last_token_pool(hidden, tok["attention_mask"], self.pool_fp32)
            else:
                pooled = _mean_pool(hidden, tok["attention_mask"], self.pool_fp32)
            if self.l2_normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            v = pooled.float().cpu().numpy()
            out.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
        return np.concatenate(out, axis=0).astype(np.float32)
