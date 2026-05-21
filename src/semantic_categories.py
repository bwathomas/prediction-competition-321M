"""Deterministic 15-category semantic assigner for item variants.

This module groups item variants from the measurement-db competition into a
fixed taxonomy of 15 *conceptual* categories (skill / modality / safety axes)
so training can be category-stratified independent of the hidden platform's
internal stratification.

The 15 categories were chosen by inspecting `items.parquet` across all 16
benchmarks; see the docstring of :data:`CATEGORY_NAMES` for the canonical
list. The assigner uses only fields available both at training time and in
the four-field runtime contract (`benchmark`, `condition`, `item_content`),
so the same bucket assignment works in `predict()` if needed.

Public API
----------

- :data:`CATEGORY_NAMES`           -- ordered list of 15 category slugs
- :data:`CATEGORY_TO_ID`           -- mapping slug -> int in ``[0, 15)``
- :func:`assign_semantic_category` -- ``row -> slug``
- :func:`assign_semantic_category_id` -- ``row -> int``
- :func:`compute_semantic_sample_weights` -- per-row training weights
- :func:`semantic_category_report`        -- diagnostic dataframe

The weight formula implemented here is

    w_lambda(i, s) = (1 / n_i) * ( |I| / (15 * |I_{c(i)}|) ) ** lambda

where ``i`` is an item-variant id, ``c(i)`` is its semantic category,
``n_i`` is the number of training rows for the item, ``|I|`` is the total
number of unique item variants, and ``|I_{c(i)}|`` is the number of unique
item variants in category ``c(i)``. ``lambda = 0`` reduces to plain
item-uniform weighting; ``lambda = 1`` makes every category contribute the
same total weight regardless of how many items it contains.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import pandas as pd  # noqa: F401  (used for type hints / runtime branches)
except Exception:  # pragma: no cover - pandas is a hard dep elsewhere
    pd = None  # type: ignore[assignment]

LOG = logging.getLogger("semantic_categories")


# ---------------------------------------------------------------------------
# Canonical category list (fixed order; do not renumber existing entries)
# ---------------------------------------------------------------------------

CATEGORY_NAMES: list[str] = [
    "open_ended_chat_creative_instruction",        # 0
    "multilingual_translation",                    # 1
    "nli_commonsense_reasoning",                   # 2
    "mcq_stem_text",                               # 3
    "mcq_humanities_social_science",               # 4
    "clinical_medical_qa",                         # 5
    "competition_olympiad_math",                   # 6
    "visual_math_reasoning",                       # 7
    "visual_qa_diagram_understanding",             # 8
    "frontier_last_exam_reasoning",                # 9
    "algorithmic_competitive_programming",         # 10
    "real_world_software_engineering",             # 11
    "tool_use_function_calling",                   # 12
    "agentic_gui_os_tasks",                        # 13
    "safety_refusal_adversarial_robustness",       # 14
]
N_CATEGORIES: int = len(CATEGORY_NAMES)
assert N_CATEGORIES == 15

CATEGORY_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CATEGORY_NAMES)}


# ---------------------------------------------------------------------------
# Field-name normalization (robust to column-name variants)
# ---------------------------------------------------------------------------

_BENCHMARK_KEYS = ("benchmark", "benchmark_id", "bench")
_CONDITION_KEYS = ("condition", "test_condition", "cond")
_ITEM_CONTENT_KEYS = ("item_content", "content", "item_text")
_ITEM_ID_KEYS = ("item_variant_id", "item_id", "raw_item_id")


def _lookup(row: Mapping[str, Any] | Any, keys: Sequence[str]) -> str:
    """Best-effort string lookup over a mapping / pd.Series / dataclass-ish row."""
    if row is None:
        return ""
    for k in keys:
        try:
            if k in row:  # type: ignore[operator]
                val = row[k]
            else:
                val = None
        except Exception:
            val = getattr(row, k, None)
        if val is None:
            continue
        try:
            if isinstance(val, float) and val != val:  # NaN
                continue
        except Exception:
            pass
        s = str(val)
        if s and s.lower() not in {"nan", "none", "null"}:
            return s
    return ""


def _norm_benchmark(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _norm_condition(s: str) -> str:
    s2 = str(s).strip().lower()
    if s2 in {"", "nan", "none", "null"}:
        return "none"
    return s2


# ---------------------------------------------------------------------------
# Regex-based content cues (compiled once at import)
# ---------------------------------------------------------------------------

# Unicode blocks for common non-Latin scripts (used to detect multilingual text).
_NON_LATIN_RE = re.compile(
    "["                                           # any of:
    "\u3040-\u30ff"                              # Japanese kana
    "\u3400-\u4dbf\u4e00-\u9fff"                 # CJK
    "\uac00-\ud7af"                              # Hangul
    "\u0600-\u06ff\u0750-\u077f"                 # Arabic
    "\u0900-\u097f"                              # Devanagari
    "\u0400-\u04ff"                              # Cyrillic
    "\u0e00-\u0e7f"                              # Thai
    "\u0590-\u05ff"                              # Hebrew
    "]"
)

_TRANSLATION_RE = re.compile(
    r"\b(translate|translation|in (?:chinese|spanish|french|german|japanese|korean|"
    r"arabic|portuguese|italian|russian|hindi|swahili|yoruba|hausa)\b)",
    re.IGNORECASE,
)

_NLI_RE = re.compile(
    r"(?:^|\b)(premise\s*[:\-]|hypothesis\s*[:\-]|"
    r"is the hypothesis entailed|"
    r"it is not possible to tell|"
    r"based on this premise|"
    r"natural language inference|"
    r"common[\s_-]?sense)",
    re.IGNORECASE,
)

_SAFETY_RE = re.compile(
    r"(jailbreak|prompt[\s_-]?injection|"
    r"\[injection_task\]|"
    r"\bdan\b(?:\s*mode)?|"
    r"ignore (?:all )?(?:previous|prior) (?:instructions|prompts)|"
    r"how (?:to|do i) (?:make|build|create|synthesize) "
    r"(?:a )?(?:bomb|explosive|weapon|virus|malware|drug)|"
    r"\bharmful\b|"
    r"illegal\s+(?:drugs|activities|weapons|substance)|"
    r"hide (?:illegal|the body|evidence)|"
    r"\bhack(?:ing)?\s+(?:into|someone|password)|"
    r"\bbypass\s+(?:safety|filters|moderation)|"
    r"pretend (?:to be|you are) (?:able|allowed)|"
    r"\brefuse\b|\brefusal\b|"
    r"unsafe\s+(?:content|response)|"
    r"\bracial\s+slur)",
    re.IGNORECASE,
)

_CODE_RE = re.compile(
    r"(```[a-z+#-]*\n|"
    r"\bdef\s+[a-zA-Z_][\w]*\s*\(|"
    r"\bclass\s+[A-Z][\w]*\s*[:\(]|"
    r"\bfunction\s+[a-zA-Z_][\w]*\s*\(|"
    r"\bpublic\s+(?:static\s+)?[a-zA-Z_<>\[\]]+\s+[a-zA-Z_][\w]*\s*\(|"
    r"\bvector<|std::|#include\s*<|"
    r"<\s*html|<\s*body|<\s*div\b|<\s*script\b|<\s*style\b|"
    r"\bSELECT\s+.+\s+FROM\b|"
    r"input\s*-\s*output|"
    r"sample\s+input|"
    r"\bleetcode\b|atcoder|codeforces)",
    re.IGNORECASE,
)

_HTML_CSS_RE = re.compile(
    r"(<\s*html|<\s*body|<\s*div\b|<\s*script\b|<\s*style\b|"
    r"\.(?:css|html)\b|css\s+(?:rule|style)|html\s+page)",
    re.IGNORECASE,
)

_MATH_RE = re.compile(
    r"(\\frac|\\sum|\\int|\\sqrt|\\mathbb|\\mathcal|"
    r"\$\$|\\\(|\\\[|"
    r"\bprove\s+that\b|\btheorem\b|\blemma\b|"
    r"\bolympiad\b|\baime\b|\busamo\b|\bputnam\b|\bimo\b|\bimc\b|miklos|"
    r"\bcompetition\s+math|"
    r"\binequality\b|\bcombinator|"
    r"\bpolynomial\b|\bdivisor|\bmodulo\b|"
    r"\\angle|\\triangle|\bdihedral\b)",
    re.IGNORECASE,
)

_VISUAL_RE = re.compile(
    r"(\bimage\b|\bphoto\b|\bpicture\b|\bfigure\b|"
    r"\bdiagram\b|\bchart\b|\bgraph\b|\babove (?:image|figure)\b|"
    r"\bin the (?:image|figure|picture|photo)\b|food\s+web|"
    r"\bsee the (?:image|figure))",
    re.IGNORECASE,
)

_MEDICAL_RE = re.compile(
    r"(\bdiagnosis\b|\bdiagnose|"
    r"\bsymptoms?\b|\bpatient\b|\bclinical\b|\btreatment\b|"
    r"\bmedication\b|\bdrug\b\s+(?:dose|interaction|allergy)|"
    r"\bhospital\b|\bphysician\b|\bnurse\b|\bsurgery\b|"
    r"\bhealth\b|\bdisease\b|\bsyndrome\b|\bpneum|\bcardiac\b|\binfection\b|"
    r"public\s+health|prescrib|antibiotic|vaccine|\bdose\b|"
    r"\boncology\b|\bpediatric\b)",
    re.IGNORECASE,
)

# MMLU-Pro-style routing keywords. Used to project the free-text item content
# to a coarse subject when the benchmark doesn't ship a subject column.
_MMLUPRO_MATH_RE = re.compile(
    r"\b(math|mathematics|algebra|geometry|calculus|probability|statistics|"
    r"differential|integral|matrix|vector\s+space|eigen|topology|"
    r"combinatoric|number\s+theory)\b",
    re.IGNORECASE,
)
_MMLUPRO_MED_RE = re.compile(
    r"\b(medicine|medical|clinical|health|biomedical|physiology|"
    r"anatomy|pharmac|pathology|epidemiology|nursing)\b",
    re.IGNORECASE,
)
_MMLUPRO_STEM_RE = re.compile(
    r"\b(physics|chemistry|chemical|biology|biological|"
    r"engineer|engineering|computer\s+science|cs\b|algorithm|"
    r"thermodynam|electromagn|quantum|molecule|reaction|cell|protein|gene|"
    r"convect|circuit|cache|kernel|process|operating\s+system|set-associative)\b",
    re.IGNORECASE,
)
_MMLUPRO_HUM_RE = re.compile(
    r"\b(history|philosoph|psycholog|sociolog|law\b|legal|"
    r"business|management|economic|finance|accounting|"
    r"literature|political|government|ethic|religion|theology|"
    r"family\s+therapist|salvador\s+minuchin)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Per-benchmark routing
# ---------------------------------------------------------------------------


def _classify_ultrafeedback(content: str, condition: str) -> str:
    if _SAFETY_RE.search(content):
        return "safety_refusal_adversarial_robustness"
    if _NON_LATIN_RE.search(content) or _TRANSLATION_RE.search(content):
        return "multilingual_translation"
    if _NLI_RE.search(content):
        return "nli_commonsense_reasoning"
    if _HTML_CSS_RE.search(content):
        # HTML/CSS in ultrafeedback is closer to "real-world web SE" than
        # algorithmic problem solving; keep it in the SWE bucket to mirror
        # how it behaves at runtime (long-form, project-flavored).
        return "real_world_software_engineering"
    if _CODE_RE.search(content):
        return "algorithmic_competitive_programming"
    if _MATH_RE.search(content):
        return "competition_olympiad_math"
    return "open_ended_chat_creative_instruction"


def _classify_rewardbench(content: str, condition: str) -> str:
    if _SAFETY_RE.search(content):
        return "safety_refusal_adversarial_robustness"
    if _CODE_RE.search(content):
        return "algorithmic_competitive_programming"
    if _MATH_RE.search(content):
        return "competition_olympiad_math"
    if _MEDICAL_RE.search(content):
        return "clinical_medical_qa"
    return "open_ended_chat_creative_instruction"


def _classify_mmlupro(content: str, condition: str) -> str:
    if _MMLUPRO_MATH_RE.search(content) or _MATH_RE.search(content):
        return "competition_olympiad_math"
    if _MMLUPRO_MED_RE.search(content) or _MEDICAL_RE.search(content):
        return "clinical_medical_qa"
    if _MMLUPRO_STEM_RE.search(content):
        return "mcq_stem_text"
    if _MMLUPRO_HUM_RE.search(content):
        return "mcq_humanities_social_science"
    # Default fall-through: humanities/social science covers the long tail
    # of "other" / "psychology" / "law" rows we'd otherwise misclassify.
    if _NLI_RE.search(content):
        return "nli_commonsense_reasoning"
    return "mcq_humanities_social_science"


def _classify_agentdojo(content: str, condition: str) -> str:
    if "[injection_task]" in content.lower() or _SAFETY_RE.search(content):
        return "safety_refusal_adversarial_robustness"
    return "tool_use_function_calling"


def _classify_hle(content: str, condition: str, item_id: str) -> str:
    """HLE items in the public dump are often just stub ids (``hle:...``),
    so we cannot reliably split math from non-math from the content alone.
    Route the math-flavored stems we *can* see to category 6 and everything
    else to the frontier-reasoning bucket.
    """
    if _MATH_RE.search(content) or _MMLUPRO_MATH_RE.search(content):
        return "competition_olympiad_math"
    return "frontier_last_exam_reasoning"


def _classify_unknown(content: str, condition: str) -> str:
    if _SAFETY_RE.search(content):
        return "safety_refusal_adversarial_robustness"
    if _NON_LATIN_RE.search(content) or _TRANSLATION_RE.search(content):
        return "multilingual_translation"
    if _CODE_RE.search(content) or _HTML_CSS_RE.search(content):
        return "algorithmic_competitive_programming"
    if _MEDICAL_RE.search(content):
        return "clinical_medical_qa"
    if _MATH_RE.search(content):
        return "competition_olympiad_math"
    if _VISUAL_RE.search(content):
        return "visual_qa_diagram_understanding"
    return "open_ended_chat_creative_instruction"


# Direct benchmark -> category routing. Anything mapped here bypasses the
# content-based classifier below.
_DIRECT_BENCHMARK_ROUTE: dict[str, str] = {
    "afrimedqa": "clinical_medical_qa",
    "mathvista_mini": "visual_math_reasoning",
    "ai2d_test": "visual_qa_diagram_understanding",
    "mmbench_v11": "visual_qa_diagram_understanding",
    "livecodebench": "algorithmic_competitive_programming",
    "swebench": "real_world_software_engineering",
    "bfcl": "tool_use_function_calling",
    "androidworld": "agentic_gui_os_tasks",
    "cybench": "agentic_gui_os_tasks",
    "matharena": "competition_olympiad_math",
    "mtbench": "open_ended_chat_creative_instruction",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assign_semantic_category(row: Mapping[str, Any] | Any) -> str:
    """Return the semantic-category slug for a single row / dict.

    Required (best-effort) fields: ``benchmark``, ``item_content``.
    Optional: ``condition``, ``item_variant_id`` / ``item_id``.

    Always returns a slug from :data:`CATEGORY_NAMES`. Falls back to the
    open-ended-chat category on completely unrecognized input.
    """
    benchmark = _norm_benchmark(_lookup(row, _BENCHMARK_KEYS))
    condition = _norm_condition(_lookup(row, _CONDITION_KEYS))
    content = _lookup(row, _ITEM_CONTENT_KEYS) or ""
    item_id = _lookup(row, _ITEM_ID_KEYS) or ""

    # Treat HLE stub ids ("hle:66ea...") as part of the content for routing.
    content_for_match = content if content else item_id

    direct = _DIRECT_BENCHMARK_ROUTE.get(benchmark)
    if direct is not None:
        return direct

    if benchmark == "hle":
        return _classify_hle(content_for_match, condition, item_id)
    if benchmark == "agentdojo":
        return _classify_agentdojo(content_for_match, condition)
    if benchmark == "rewardbench":
        return _classify_rewardbench(content_for_match, condition)
    if benchmark == "mmlupro":
        return _classify_mmlupro(content_for_match, condition)
    if benchmark == "ultrafeedback":
        return _classify_ultrafeedback(content_for_match, condition)

    return _classify_unknown(content_for_match, condition)


def assign_semantic_category_id(row: Mapping[str, Any] | Any) -> int:
    """Same as :func:`assign_semantic_category` but returns the integer id."""
    return CATEGORY_TO_ID[assign_semantic_category(row)]


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def _pick_item_col(df: "pd.DataFrame", item_col: str) -> str:
    """Return a valid item-identifier column from ``df``.

    Tries ``item_col`` first, then falls back to common aliases used in this
    repository (``item_key``, ``item_id``).
    """
    if item_col in df.columns:
        return item_col
    for alt in ("item_variant_id", "item_key", "item_id"):
        if alt in df.columns:
            LOG.info(
                "compute_semantic_sample_weights: %r missing, using %r",
                item_col,
                alt,
            )
            return alt
    raise KeyError(
        f"No item-identifier column found in dataframe. Tried "
        f"{[item_col, 'item_variant_id', 'item_key', 'item_id']}; "
        f"available columns: {list(df.columns)}"
    )


def assign_semantic_categories_df(
    df: "pd.DataFrame",
    *,
    item_col: str = "item_variant_id",
    cache_per_item: bool = True,
) -> np.ndarray:
    """Vectorized category-id assignment over a dataframe.

    Returns an ``int64`` numpy array of length ``len(df)`` containing
    category ids in ``[0, 15)``. When ``cache_per_item`` is true (default),
    we assign categories once per unique item identifier and broadcast back
    to rows -- this is dramatically faster on the full 5M-row training set
    than calling :func:`assign_semantic_category_id` per row.
    """
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for assign_semantic_categories_df")

    if len(df) == 0:
        return np.zeros(0, dtype=np.int64)

    item_col = _pick_item_col(df, item_col)

    if not cache_per_item:
        return np.fromiter(
            (assign_semantic_category_id(r) for r in df.to_dict(orient="records")),
            dtype=np.int64,
            count=len(df),
        )

    # Per-item: pick the first row for each item and classify it.
    unique = df.drop_duplicates(subset=[item_col])
    cat_for_item: dict[str, int] = {}
    for r in unique.to_dict(orient="records"):
        cat_for_item[str(r[item_col])] = assign_semantic_category_id(r)
    ids = df[item_col].astype(str).map(cat_for_item).to_numpy()
    # Map should be total by construction; guard against NaNs anyway.
    ids = np.where(pd.isna(ids), 0, ids).astype(np.int64)
    return ids


def compute_semantic_sample_weights(
    df: "pd.DataFrame",
    lambda_: float = 1.0,
    *,
    item_col: str = "item_variant_id",
    normalize: bool = True,
    return_report: bool = False,
) -> "np.ndarray | tuple[np.ndarray, dict[str, Any]]":
    """Compute per-row training weights with semantic-category stratification.

    The formula is

        w_lambda(i, s) = (1 / n_i) * ( |I| / (15 * |I_{c(i)}|) ) ** lambda_

    where ``i`` is the item variant of row ``s``, ``n_i`` is the number of
    rows for that item in ``df``, ``c(i)`` is the semantic category of item
    ``i``, ``|I_{c(i)}|`` is the number of *unique items* in that category,
    and ``|I|`` is the total number of unique items.

    Parameters
    ----------
    df :
        DataFrame indexed by row. Must contain an item-identifier column
        (``item_variant_id`` by default; falls back to ``item_key`` /
        ``item_id``) plus the runtime fields used by the category assigner.
    lambda_ :
        Strength of the category-stratification term. ``0`` -> pure
        item-uniform weights. ``1`` -> every category contributes the same
        total weight regardless of size.
    normalize :
        If true (default), rescale weights to mean 1.0. Required to keep
        the effective learning rate comparable to the unweighted run.
    return_report :
        If true, return ``(weights, report)`` where ``report`` is a dict
        suitable for logging (per-category item counts, row counts, total
        weight, etc.).

    Returns
    -------
    weights : np.ndarray
        ``[N]`` float32 array of per-row weights.
    report : dict (optional)
        See :func:`semantic_category_report`.
    """
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for compute_semantic_sample_weights")

    n_rows = len(df)
    if n_rows == 0:
        empty = np.zeros(0, dtype=np.float32)
        if return_report:
            return empty, {"n_rows": 0, "n_items": 0, "by_category": {}}
        return empty

    lambda_f = float(lambda_)
    item_col = _pick_item_col(df, item_col)
    items = df[item_col].astype(str).to_numpy()

    cat_ids = assign_semantic_categories_df(df, item_col=item_col, cache_per_item=True)

    # n_i: rows per item
    item_index, inverse = np.unique(items, return_inverse=True)
    item_row_counts = np.bincount(inverse, minlength=len(item_index)).astype(np.float64)
    item_row_counts = np.maximum(item_row_counts, 1.0)
    n_i_per_row = item_row_counts[inverse]

    # |I| and |I_c|: count UNIQUE items, not rows
    # Map each item to its category (first row's category, which is stable
    # because cache_per_item is on).
    first_row_for_item = np.zeros(len(item_index), dtype=np.int64)
    # np.unique with return_index would give us this in one call:
    _, first_row_for_item = np.unique(items, return_index=True)
    item_cat_ids = cat_ids[first_row_for_item]
    items_per_cat = np.bincount(item_cat_ids, minlength=N_CATEGORIES).astype(np.float64)
    total_items = float(len(item_index))

    safe_items_per_cat = np.maximum(items_per_cat, 1.0)
    cat_ratio = total_items / (N_CATEGORIES * safe_items_per_cat)
    cat_factor = np.power(cat_ratio, lambda_f)

    # Empty categories get factor 0; they will never appear in cat_ids[]
    # broadcast below, but be safe.
    cat_factor = np.where(items_per_cat > 0, cat_factor, 0.0)

    per_row_cat_factor = cat_factor[cat_ids]
    w = (1.0 / n_i_per_row) * per_row_cat_factor
    w = np.asarray(w, dtype=np.float64)

    if not np.all(np.isfinite(w)):
        LOG.warning(
            "compute_semantic_sample_weights: %d non-finite weights -> 0",
            int((~np.isfinite(w)).sum()),
        )
        w = np.where(np.isfinite(w), w, 0.0)

    if normalize:
        mean_w = float(w.mean()) if len(w) else 0.0
        if mean_w > 0:
            w = w / mean_w

    w32 = w.astype(np.float32)

    if not return_report:
        return w32

    report = {
        "n_rows": int(n_rows),
        "n_items": int(total_items),
        "lambda_": lambda_f,
        "normalize": bool(normalize),
        "mean_weight": float(w32.mean()) if len(w32) else 0.0,
        "min_weight": float(w32.min()) if len(w32) else 0.0,
        "max_weight": float(w32.max()) if len(w32) else 0.0,
        "by_category": _per_category_summary(
            items=items,
            cat_ids=cat_ids,
            item_index=item_index,
            inverse=inverse,
            item_cat_ids=item_cat_ids,
            row_weights=w32,
        ),
    }
    return w32, report


def _per_category_summary(
    *,
    items: np.ndarray,
    cat_ids: np.ndarray,
    item_index: np.ndarray,
    inverse: np.ndarray,
    item_cat_ids: np.ndarray,
    row_weights: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Build a per-category {name: {n_items, n_rows, total_weight}} dict."""
    out: dict[str, dict[str, float]] = {}
    n_items_per_cat = np.bincount(item_cat_ids, minlength=N_CATEGORIES)
    n_rows_per_cat = np.bincount(cat_ids, minlength=N_CATEGORIES)
    total_w_per_cat = np.zeros(N_CATEGORIES, dtype=np.float64)
    # Accumulate row weights per category in a single pass.
    np.add.at(total_w_per_cat, cat_ids, row_weights.astype(np.float64))
    for cid, name in enumerate(CATEGORY_NAMES):
        out[name] = {
            "n_items": int(n_items_per_cat[cid]),
            "n_rows": int(n_rows_per_cat[cid]),
            "total_weight": float(total_w_per_cat[cid]),
        }
    return out


def semantic_category_report(
    df: "pd.DataFrame",
    *,
    item_col: str = "item_variant_id",
    lambda_: float = 1.0,
    normalize: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper: build a report without keeping the weight array."""
    _, report = compute_semantic_sample_weights(
        df,
        lambda_=lambda_,
        item_col=item_col,
        normalize=normalize,
        return_report=True,
    )  # type: ignore[misc]
    return report


def format_semantic_category_report(report: Mapping[str, Any]) -> str:
    """Pretty-print a report as a multi-line table for logs / notebooks."""
    lines = [
        f"Semantic categories | n_rows={report.get('n_rows', 0):,} "
        f"n_items={report.get('n_items', 0):,} "
        f"lambda={report.get('lambda_', 'n/a')} "
        f"normalize={report.get('normalize', 'n/a')}",
        f"  mean_w={report.get('mean_weight', 0):.4f} "
        f"min_w={report.get('min_weight', 0):.4f} "
        f"max_w={report.get('max_weight', 0):.4f}",
        f"  {'category':40s} {'n_items':>10s} {'n_rows':>12s} {'tot_weight':>12s}",
    ]
    by_cat = report.get("by_category", {})
    for name in CATEGORY_NAMES:
        stats = by_cat.get(name, {})
        lines.append(
            f"  {name:40s} "
            f"{int(stats.get('n_items', 0)):>10d} "
            f"{int(stats.get('n_rows', 0)):>12d} "
            f"{float(stats.get('total_weight', 0.0)):>12.2f}"
        )
    return "\n".join(lines)


__all__ = [
    "CATEGORY_NAMES",
    "CATEGORY_TO_ID",
    "N_CATEGORIES",
    "assign_semantic_categories_df",
    "assign_semantic_category",
    "assign_semantic_category_id",
    "compute_semantic_sample_weights",
    "format_semantic_category_report",
    "semantic_category_report",
]
