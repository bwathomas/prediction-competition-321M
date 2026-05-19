"""Sanity checks: data, embeddings, and models.

Each check returns a `CheckResult`. The notebook collects them all into a
single dataframe so it's obvious at a glance which invariants passed.

Conventions:
- `ok=True` means PASSED.
- `level` is "fail" if the check should block training, "warn" if it's just
  diagnostic.
- `details` is small enough to print in a notebook cell.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .data import REQUIRED_RUNTIME_FIELDS, stable_sha256
from .embeddings import TransformerEmbedder

LOG = logging.getLogger("sanity")


# ---------------------------------------------------------------------------
# Result type and registry
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    ok: bool
    level: str = "fail"            # fail | warn
    details: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "level": self.level,
            **{f"detail_{k}": v for k, v in self.details.items()},
        }


def _format(results: list[CheckResult]) -> pd.DataFrame:
    rows = [r.to_row() for r in results]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data sanity
# ---------------------------------------------------------------------------


def check_required_columns(df: pd.DataFrame) -> CheckResult:
    missing = [c for c in REQUIRED_RUNTIME_FIELDS if c not in df.columns]
    return CheckResult(
        name="data.required_columns",
        ok=not missing,
        details={"missing": missing, "n_rows": len(df)},
    )


def check_label_binary_or_unit_interval(df: pd.DataFrame) -> CheckResult:
    y = pd.to_numeric(df["label"], errors="coerce")
    n_nan = int(y.isna().sum())
    out_of_unit = int(((y < 0.0) | (y > 1.0)).sum())
    distinct = int(y.dropna().round(6).nunique())
    return CheckResult(
        name="data.label_in_unit_interval",
        ok=(n_nan == 0 and out_of_unit == 0),
        details={"nan": n_nan, "out_of_unit": out_of_unit, "distinct": distinct},
        level="warn",
    )


def check_missing_values(df: pd.DataFrame) -> CheckResult:
    counts = df.isna().sum().to_dict()
    bad = {k: int(v) for k, v in counts.items() if v > 0}
    return CheckResult(
        name="data.missing_values",
        ok=not bad,
        level="warn",
        details=bad,
    )


def check_condition_normalization(df: pd.DataFrame) -> CheckResult:
    bad = (
        df["condition"]
        .astype(str)
        .str.lower()
        .isin({"", "nan", "null"})
        .sum()
    )
    return CheckResult(
        name="data.condition_normalized",
        ok=int(bad) == 0,
        details={"bad_rows": int(bad)},
    )


def check_duplicate_rows(df: pd.DataFrame) -> CheckResult:
    dup = int(df.duplicated(subset=["subject_key", "item_key", "label"]).sum())
    return CheckResult(
        name="data.no_exact_duplicate_rows",
        ok=dup == 0,
        level="warn",
        details={"duplicate_rows": dup},
    )


def check_inconsistent_pair_labels(df: pd.DataFrame) -> CheckResult:
    grp = df.groupby(["subject_key", "item_key"])["label"]
    bad_pairs = int((grp.nunique() > 1).sum())
    return CheckResult(
        name="data.consistent_pair_labels",
        ok=bad_pairs == 0,
        level="warn",
        details={"inconsistent_pairs": bad_pairs},
    )


def check_item_cold_start_leakage(
    train: pd.DataFrame, val: pd.DataFrame
) -> CheckResult:
    train_items = set(train["item_key"])
    val_items = set(val["item_key"])
    overlap = train_items & val_items
    return CheckResult(
        name="data.item_cold_start_leakage",
        ok=not overlap,
        details={
            "n_overlap": len(overlap),
            "n_train_items": len(train_items),
            "n_val_items": len(val_items),
        },
    )


def check_subject_overlap(train: pd.DataFrame, val: pd.DataFrame) -> CheckResult:
    """In item cold-start, val subjects SHOULD appear in train (warn only)."""
    train_subjects = set(train["subject_key"])
    val_subjects = set(val["subject_key"])
    unseen = val_subjects - train_subjects
    return CheckResult(
        name="data.val_subjects_in_train",
        ok=not unseen,
        level="warn",
        details={"n_val_subjects_unseen": len(unseen)},
    )


def check_key_stability(df: pd.DataFrame, n: int = 50) -> CheckResult:
    """item_key must equal sha256(benchmark + condition + item_content)."""
    sample = df.head(n)
    rebuilt = [
        stable_sha256(b, c, t)
        for b, c, t in zip(
            sample["benchmark"], sample["condition"], sample["item_content"]
        )
    ]
    mismatches = int((sample["item_key"].values != np.array(rebuilt)).sum())
    return CheckResult(
        name="data.item_key_stable",
        ok=mismatches == 0,
        details={"checked": int(min(n, len(df))), "mismatches": mismatches},
    )


# ---------------------------------------------------------------------------
# Embedding sanity
# ---------------------------------------------------------------------------


def check_embedding_shape(
    embedder: TransformerEmbedder, expected_dim: int | None = None
) -> CheckResult:
    dim = embedder.embedding_dim
    ok = dim > 0 and (expected_dim is None or dim == expected_dim)
    return CheckResult(
        name="embed.shape",
        ok=ok,
        details={"dim": int(dim), "expected": expected_dim},
    )


def check_embedding_determinism(embedder: TransformerEmbedder) -> CheckResult:
    """Same text -> same key -> same embedding (within float tolerance)."""
    import hashlib

    text = "deterministic sanity probe"
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out_a, _ = embedder.embed_unique(kind="item", keys=[key], texts=[text])
    # Second call should hit the in-memory cache and return the same vector.
    out_b, _ = embedder.embed_unique(kind="item", keys=[key], texts=[text])
    a = out_a[key]
    b = out_b[key]
    diff = float(np.max(np.abs(a - b)))
    return CheckResult(
        name="embed.same_text_same_embedding",
        ok=diff < 1e-5,
        level="warn",
        details={"max_abs_diff": diff},
    )


def check_embedding_nan_inf(emb: np.ndarray) -> CheckResult:
    n_nan = int((~np.isfinite(emb)).sum())
    n_zero = int((np.linalg.norm(emb, axis=1) < 1e-8).sum())
    return CheckResult(
        name="embed.nan_or_zero_norm",
        ok=(n_nan == 0 and n_zero == 0),
        details={"nan_or_inf": n_nan, "zero_norm_rows": n_zero},
    )


def check_embedding_truncation(stats: Mapping) -> CheckResult:
    rate = float(stats.get("truncation_rate", 0.0))
    return CheckResult(
        name="embed.truncation_rate",
        ok=rate < 0.50,
        level="warn",
        details={"rate": rate, "p95_tokens": stats.get("p95_tokens"), "max_tokens": stats.get("max_tokens")},
    )


# ---------------------------------------------------------------------------
# Model sanity
# ---------------------------------------------------------------------------


def check_forward_pass(
    model: torch.nn.Module,
    *,
    item_emb_dim: int,
    n_subjects: int,
    n_bc: int,
    subject_emb_dim: int = 0,
    device: str | None = None,
) -> CheckResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    B = 8
    s = torch.randint(0, max(1, n_subjects), (B,), device=device)
    bc = torch.randint(0, max(1, n_bc), (B,), device=device)
    ie = torch.randn(B, item_emb_dim, device=device)
    se = torch.randn(B, subject_emb_dim, device=device) if subject_emb_dim > 0 else None
    with torch.inference_mode():
        logits = model(s, bc, ie, se)
        prob = torch.sigmoid(logits)
    finite_logits = bool(torch.isfinite(logits).all().item())
    in_unit = bool(((prob >= 0) & (prob <= 1)).all().item())
    return CheckResult(
        name="model.forward_pass",
        ok=(logits.shape == (B,) and finite_logits and in_unit),
        details={
            "shape": list(logits.shape),
            "finite_logits": finite_logits,
            "prob_in_unit": in_unit,
            "native_float": isinstance(float(prob[0].item()), float),
        },
    )


def check_overfit_tiny_batch(
    model: torch.nn.Module,
    *,
    item_emb_dim: int,
    n_subjects: int,
    n_bc: int,
    subject_emb_dim: int = 0,
    n: int = 256,
    steps: int = 300,
    lr: float = 5e-2,
    device: str | None = None,
    target_loss: float = 0.05,
) -> CheckResult:
    """Can the model drive BCE down on a small held set? If not, it's broken."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()
    rng = torch.Generator(device="cpu").manual_seed(0)
    s = torch.randint(0, max(1, n_subjects), (n,), generator=rng)
    bc = torch.randint(0, max(1, n_bc), (n,), generator=rng)
    ie = torch.randn(n, item_emb_dim, generator=rng)
    se = torch.randn(n, subject_emb_dim, generator=rng) if subject_emb_dim > 0 else None
    y = torch.bernoulli(torch.sigmoid(torch.randn(n, generator=rng) * 1.5))
    s, bc, ie, y = [t.to(device) for t in (s, bc, ie, y)]
    if se is not None:
        se = se.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    final = float("nan")
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(s, bc, ie, se)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        final = float(loss.item())
    return CheckResult(
        name="model.overfit_tiny_batch",
        ok=final < target_loss,
        level="warn",
        details={"final_loss": final, "target_loss": target_loss},
    )


def check_random_labels_sanity(
    model_factory,
    *,
    item_emb_dim: int,
    n_subjects: int,
    n_bc: int,
    n_train: int = 1024,
    n_val: int = 512,
    epochs: int = 5,
    device: str | None = None,
) -> CheckResult:
    """Train on random labels; validation log-loss must stay ~log(2).

    A model that achieves much better than ~log(2) on random labels is
    leaking through indices or embeddings somehow.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device).train()
    rng = torch.Generator(device="cpu").manual_seed(1)
    s = torch.randint(0, max(1, n_subjects), (n_train + n_val,), generator=rng)
    bc = torch.randint(0, max(1, n_bc), (n_train + n_val,), generator=rng)
    ie = torch.randn(n_train + n_val, item_emb_dim, generator=rng)
    y = torch.bernoulli(torch.full((n_train + n_val,), 0.5))
    s, bc, ie, y = [t.to(device) for t in (s, bc, ie, y)]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(s[:n_train], bc[:n_train], ie[:n_train])
        loss = loss_fn(logits, y[:n_train])
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        val_logits = model(s[n_train:], bc[n_train:], ie[n_train:])
        val_prob = torch.sigmoid(val_logits).clamp(1e-6, 1 - 1e-6)
        ll = -(
            y[n_train:] * val_prob.log() + (1.0 - y[n_train:]) * (1.0 - val_prob).log()
        ).mean()
    return CheckResult(
        name="model.random_labels_log_loss",
        ok=float(ll.item()) > 0.55,
        level="warn",
        details={"val_log_loss": float(ll.item()), "expected_around": 0.693},
    )


def check_ablate_item_embeddings(
    model: torch.nn.Module,
    val_inputs: tuple,
    *,
    device: str | None = None,
) -> CheckResult:
    """Replace item embeddings with zeros; report the log-loss delta."""
    s_idx, bc_idx, item_emb, subject_emb, y = val_inputs
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.inference_mode():
        base = torch.sigmoid(
            model(s_idx.to(device), bc_idx.to(device), item_emb.to(device),
                  subject_emb.to(device) if subject_emb is not None and subject_emb.shape[-1] > 0 else None)
        ).cpu().clamp(1e-6, 1 - 1e-6)
        zeroed = torch.sigmoid(
            model(s_idx.to(device), bc_idx.to(device), torch.zeros_like(item_emb).to(device),
                  subject_emb.to(device) if subject_emb is not None and subject_emb.shape[-1] > 0 else None)
        ).cpu().clamp(1e-6, 1 - 1e-6)
    ll = -(y * base.log() + (1 - y) * (1 - base).log()).mean()
    ll_zero = -(y * zeroed.log() + (1 - y) * (1 - zeroed).log()).mean()
    return CheckResult(
        name="model.ablate_item_embeddings",
        ok=float(ll_zero) > float(ll),  # zeroing should hurt
        level="warn",
        details={"base_ll": float(ll), "zeroed_item_ll": float(ll_zero)},
    )


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def run_data_checks(
    df: pd.DataFrame,
    *,
    train: pd.DataFrame | None = None,
    val: pd.DataFrame | None = None,
) -> list[CheckResult]:
    results = [
        check_required_columns(df),
        check_label_binary_or_unit_interval(df),
        check_missing_values(df),
        check_condition_normalization(df),
        check_duplicate_rows(df),
        check_inconsistent_pair_labels(df),
        check_key_stability(df),
    ]
    if train is not None and val is not None:
        results.append(check_item_cold_start_leakage(train, val))
        results.append(check_subject_overlap(train, val))
    return results


def to_dataframe(results: Iterable[CheckResult]) -> pd.DataFrame:
    return _format(list(results))


def print_results(results: Iterable[CheckResult]) -> None:
    for r in results:
        flag = "PASS" if r.ok else ("FAIL" if r.level == "fail" else "WARN")
        print(f"[{flag}] {r.name:40s} {r.details}")


__all__ = [
    "CheckResult",
    "check_ablate_item_embeddings",
    "check_condition_normalization",
    "check_duplicate_rows",
    "check_embedding_determinism",
    "check_embedding_nan_inf",
    "check_embedding_shape",
    "check_embedding_truncation",
    "check_forward_pass",
    "check_inconsistent_pair_labels",
    "check_item_cold_start_leakage",
    "check_key_stability",
    "check_label_binary_or_unit_interval",
    "check_missing_values",
    "check_overfit_tiny_batch",
    "check_random_labels_sanity",
    "check_required_columns",
    "check_subject_overlap",
    "print_results",
    "run_data_checks",
    "to_dataframe",
]
