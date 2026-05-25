"""Smoke test: validate the pandas StringDtype shim.

Tests the shim two ways:

1. **Direct path**: if the current pandas already accepts the 2-arg
   signature, the shim should be a no-op.

2. **Simulated old pandas**: we monkey-patch ``pandas.StringDtype.__init__``
   to a fake old signature that only accepts ``(self, storage)``, then
   install the shim and verify that ``pd.StringDtype('python', pd.NA)``
   now succeeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

print("pandas version:", pd.__version__)

# --- Test 1: no-op on modern pandas -----------------------------------------
from src.ensemble_helpers import _install_pandas_stringdtype_shim  # noqa: E402

installed = _install_pandas_stringdtype_shim()
print("shim needed on this pandas?", installed)
try:
    sd = pd.StringDtype("python", pd.NA)
    print("modern signature works:", sd)
except TypeError as e:
    print("modern signature broken:", e)
    raise SystemExit(1)

# --- Test 2: simulate old pandas and re-install shim ------------------------
print("\n--- simulating old pandas (1-arg StringDtype) ---")
StringDtype = pd.StringDtype
real_init = StringDtype.__init__


def fake_old_init(self, storage=None):  # noqa: D401
    """Pretend to be pandas 2.2: only accepts ``storage``."""
    return real_init(self, storage)


# Force-replace with the "old" signature.
StringDtype.__init__ = fake_old_init  # type: ignore[method-assign]
print(
    "after fake old patch, 2-arg call should fail:",
    end=" ",
)
try:
    pd.StringDtype("python", pd.NA)
    print("UNEXPECTED OK")
    raise SystemExit(1)
except TypeError as e:
    print("got expected TypeError:", e)

# Now install our shim and try again.
ok = _install_pandas_stringdtype_shim()
print("shim installed under simulated old pandas?", ok)
try:
    sd = pd.StringDtype("python", pd.NA)
    print("post-shim 2-arg call:", sd)
except TypeError as e:
    print("shim FAILED:", e)
    raise SystemExit(1)

# Verify shim is idempotent: calling again is a no-op.
ok2 = _install_pandas_stringdtype_shim()
print("idempotent re-install:", ok2)

# Restore real init so we don't poison subsequent imports.
StringDtype.__init__ = real_init  # type: ignore[method-assign]
print("restored real __init__; sanity:", pd.StringDtype("python", pd.NA))
print("\nOK")
