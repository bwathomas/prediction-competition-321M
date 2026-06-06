"""Submission roster — the locked, retrained 3-family ensemble spec (P0).

This module is **offline ship-pipeline code**, not runtime code: it may import
``yaml`` / stdlib freely (the Codabench runtime never imports it). It loads
``configs/submission_roster.yaml`` and exposes it as typed data:

    >>> from scripts.ship.roster import get_roster, frozen_meta_columns
    >>> roster = get_roster("qwen")           # list[MemberSpec], frozen order
    >>> [m.key for m in roster]
    ['lgb', 'lgb_goss', 'lgb_dart', 'xgb', 'cat', 'ExtraTrees', 'mlp', 'fm', 'irt', 'featroute']
    >>> frozen_meta_columns()[:2]
    ['qwen.lgb', 'qwen.lgb_goss']

Why this is DATA, not code
--------------------------
The roster *order* is the meta-input column order: when the LightGBM
cross_entropy stacker (Layer-2) is fit on the ``3 x n_members`` honest-OOF
matrix, column ``j`` is produced by a specific (family, member) pair. If the
roster order drifts between the OOF-generation run and the shipped refit run,
the frozen meta is applied to permuted columns and the submission silently
degrades. Declaring the order here once, as data, and freezing it via
``frozen_meta_columns()`` is the guard. ``tests/test_roster.py`` asserts the
order is stable and the keys unique.

The per-family member list is sourced verbatim from the Explore-confirmed final
AIDE rosters in ``docs/OVERNIGHT_WORKING_MEMORY.md``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

# ----------------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------------
# scripts/ship/roster.py -> repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROSTER_YAML = _REPO_ROOT / "configs" / "submission_roster.yaml"

# Canonical family order. The frozen meta-column vector concatenates families in
# THIS order; do not re-order. These are the ship-pipeline (encoder-vendor)
# names; AIDE working-memory names (qwen/mistral/llama) alias onto them.
FAMILY_ORDER: tuple[str, ...] = ("qwen", "nemotron", "lgai")


# ----------------------------------------------------------------------------
# Typed data
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class BagConfig:
    """3x bagging config for a member (gap (b))."""

    n_bags: int
    bag_kind: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_bags != len(self.seeds):
            raise ValueError(
                f"n_bags={self.n_bags} but {len(self.seeds)} seeds supplied"
            )


@dataclass(frozen=True)
class MemberSpec:
    """One Layer-1 base learner in a family's frozen roster.

    Attributes
    ----------
    key
        Stable member identifier, unique within a family (e.g. ``"lgb_goss"``).
        Used to build the meta column name ``f"{family}.{key}"``.
    kind
        Member archetype (``gbdt`` / ``xgb`` / ``catboost`` / ``forest`` /
        ``knn`` / ``mlp`` / ``fwfm`` / ``irt`` / ``featroute``). Selects the
        ``*MemberState`` contract.
    fit_ref
        Dotted ``"module.function"`` reference to the fit function
        (e.g. ``"src.gbdt_member.fit_gbdt_member"``). Resolve via
        :func:`resolve_fit_fn`.
    family
        The family this member belongs to (ship-pipeline name).
    params
        Default hyperparameters (merged from the YAML ``_defaults`` block and
        any per-member overrides).
    bag
        Per-member bagging config.
    boosting
        For ``gbdt`` members, the LightGBM boosting variant
        (``gbdt`` / ``goss`` / ``dart``). ``None`` for non-gbdt members.
    pending_module
        ``True`` when ``fit_ref`` targets a module not yet authored (P2
        deliverables ``src.irt_member`` / ``src.featroute_member``). The P0
        gate validates the reference is well-formed but does not import it.
    """

    key: str
    kind: str
    fit_ref: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    bag: BagConfig | None = None
    boosting: str | None = None
    pending_module: bool = False

    @property
    def meta_column(self) -> str:
        """Name of this member's column in the stacker feature matrix."""
        return f"{self.family}.{self.key}"

    @property
    def module_path(self) -> str:
        """The ``module`` half of ``fit_ref`` (before the final dot)."""
        return self.fit_ref.rsplit(".", 1)[0]

    @property
    def fn_name(self) -> str:
        """The ``function`` half of ``fit_ref`` (after the final dot)."""
        return self.fit_ref.rsplit(".", 1)[1]


@dataclass(frozen=True)
class FamilyRoster:
    """A family's encoder + ordered member list."""

    family: str
    encoder_slug: str
    members: tuple[MemberSpec, ...]

    def keys(self) -> list[str]:
        return [m.key for m in self.members]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _load_yaml() -> dict[str, Any]:
    if not _ROSTER_YAML.exists():
        raise FileNotFoundError(f"roster config not found: {_ROSTER_YAML}")
    with _ROSTER_YAML.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"roster config is not a mapping: {_ROSTER_YAML}")
    return cfg


def _resolve_family_name(family: str, cfg: dict[str, Any]) -> str:
    """Map an alias (qwen/mistral/llama or vendor name) to the canonical name."""
    aliases: dict[str, str] = dict(cfg.get("aliases", {}))
    canonical = aliases.get(family, family)
    if canonical not in cfg.get("families", {}):
        valid = sorted(set(aliases) | set(cfg.get("families", {})))
        raise KeyError(f"unknown family {family!r}; valid: {valid}")
    return canonical


def _bag_config(cfg: dict[str, Any]) -> BagConfig:
    bd = cfg.get("bag_defaults", {})
    seeds = tuple(int(s) for s in bd.get("seeds", (0, 1, 2)))
    return BagConfig(
        n_bags=int(bd.get("n_bags", len(seeds))),
        bag_kind=str(bd.get("bag_kind", "seed")),
        seeds=seeds,
    )


def get_roster(family: str) -> list[MemberSpec]:
    """Return the frozen, ordered member list for ``family``.

    ``family`` accepts both the ship-pipeline vendor name (``qwen`` /
    ``nemotron`` / ``lgai``) and the AIDE working-memory name (``qwen`` /
    ``mistral`` / ``llama``); both resolve to the same roster.
    """
    cfg = _load_yaml()
    canonical = _resolve_family_name(family, cfg)
    bag = _bag_config(cfg)
    fam_block = cfg["families"][canonical]
    members: list[MemberSpec] = []
    for raw in fam_block["members"]:
        members.append(
            MemberSpec(
                key=str(raw["key"]),
                kind=str(raw["kind"]),
                fit_ref=str(raw["fit_ref"]),
                family=canonical,
                params=dict(raw.get("params", {}) or {}),
                bag=bag,
                boosting=(str(raw["boosting"]) if raw.get("boosting") else None),
                pending_module=bool(raw.get("pending_module", False)),
            )
        )
    return members


def get_family_roster(family: str) -> FamilyRoster:
    """Like :func:`get_roster` but also carries the encoder slug."""
    cfg = _load_yaml()
    canonical = _resolve_family_name(family, cfg)
    encoder = cfg.get("encoders", {}).get(canonical)
    if not encoder:
        raise KeyError(f"no encoder slug declared for family {canonical!r}")
    return FamilyRoster(
        family=canonical,
        encoder_slug=str(encoder),
        members=tuple(get_roster(canonical)),
    )


def encoder_slug(family: str) -> str:
    """Return the HuggingFace encoder model id for ``family``."""
    return get_family_roster(family).encoder_slug


def frozen_meta_columns() -> list[str]:
    """The frozen Layer-2 meta-input column order: ``f"{family}.{key}"``.

    Concatenated across families in :data:`FAMILY_ORDER`, members in roster
    order. This list IS the contract between the OOF-generation run and the
    shipped frozen meta — its order must never change once submissions are
    assessed against it.
    """
    cols: list[str] = []
    for fam in FAMILY_ORDER:
        for spec in get_roster(fam):
            cols.append(spec.meta_column)
    return cols


def resolve_fit_fn(spec: MemberSpec) -> Callable[..., Any]:
    """Import and return the fit function referenced by ``spec.fit_ref``.

    Raises ``ModuleNotFoundError`` for a ``pending_module`` spec whose P2
    module has not been authored yet — call sites that only need the *shape*
    of the roster (e.g. the P0 gate) should not call this on pending specs.
    """
    module = importlib.import_module(spec.module_path)
    try:
        return getattr(module, spec.fn_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ImportError(
            f"module {spec.module_path!r} has no attribute {spec.fn_name!r} "
            f"(fit_ref={spec.fit_ref!r})"
        ) from exc


def all_families() -> tuple[str, ...]:
    """Canonical family names, in frozen order."""
    return FAMILY_ORDER


if __name__ == "__main__":  # pragma: no cover - manual inspection
    for fam in FAMILY_ORDER:
        fr = get_family_roster(fam)
        print(f"{fam}  ({fr.encoder_slug})  [{len(fr.members)} members]")
        for m in fr.members:
            tag = "  (pending)" if m.pending_module else ""
            boost = f"  boosting={m.boosting}" if m.boosting else ""
            print(f"    {m.key:<11} {m.kind:<10} {m.fit_ref}{boost}{tag}")
    print(f"\nfrozen meta columns ({len(frozen_meta_columns())}):")
    print("  " + ", ".join(frozen_meta_columns()))
