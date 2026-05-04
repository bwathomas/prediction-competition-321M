"""Load a participant submission as importable modules and reset between rounds.

The platform spec says module-level code runs ONCE per container, and many
predict() calls follow. To simulate fresh containers across rounds we use
importlib.reload to reset module-level state.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class Submission:
    """Wraps a folder containing model.py (and optionally labeling.py).

    Parameters
    ----------
    submission_dir : path to the folder containing model.py
    require_labeling : if True, raise if labeling.py is missing
    model_module_name : module file basename without .py (default "model")
    labeling_module_name : module file basename without .py (default "labeling")
    """

    def __init__(
        self,
        submission_dir: str | Path,
        *,
        require_labeling: bool = False,
        model_module_name: str = "model",
        labeling_module_name: str = "labeling",
    ) -> None:
        self.submission_dir = Path(submission_dir).resolve()
        if not self.submission_dir.is_dir():
            raise FileNotFoundError(f"Submission dir not found: {self.submission_dir}")
        self.model_module_name = model_module_name
        self.labeling_module_name = labeling_module_name
        self.require_labeling = require_labeling

        self._unique_prefix = f"_subm_{abs(hash(str(self.submission_dir))) & 0xFFFFFFFF:08x}"
        self.model: ModuleType | None = None
        self.labeling: ModuleType | None = None
        self.reset()

    @property
    def _model_full_name(self) -> str:
        return f"{self._unique_prefix}_{self.model_module_name}"

    @property
    def _labeling_full_name(self) -> str:
        return f"{self._unique_prefix}_{self.labeling_module_name}"

    def _load(self, basename: str, full_name: str) -> ModuleType | None:
        path = self.submission_dir / f"{basename}.py"
        if not path.exists():
            return None
        sys.modules.pop(full_name, None)
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module

    def reset(self) -> None:
        """Reload model and labeling modules to simulate a fresh container."""
        self.model = self._load(self.model_module_name, self._model_full_name)
        if self.model is None:
            raise FileNotFoundError(
                f"{self.model_module_name}.py not found in {self.submission_dir}"
            )
        if not hasattr(self.model, "predict"):
            raise AttributeError(
                f"{self.model_module_name}.py must define predict(input, labeled)"
            )
        self.labeling = self._load(self.labeling_module_name, self._labeling_full_name)
        if self.require_labeling and self.labeling is None:
            raise FileNotFoundError(
                f"{self.labeling_module_name}.py not found in {self.submission_dir} "
                "but require_labeling=True"
            )
