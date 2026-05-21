"""Colab-visible tqdm replacement.

``tqdm.auto`` picks ``tqdm.notebook`` inside Jupyter / Colab, which renders
through ipywidgets. That works for a single top-level progress bar but
silently breaks for nested bars in Colab (the inner widget never appears,
or appears once and then never updates -- so a long training cell looks
hung from the user's perspective).

This module ships a self-contained ``ColabVisibleTqdm`` that renders as
plain HTML using ``IPython.display.update``. It supports the subset of
the tqdm API the trainer actually uses (``set_description``,
``set_postfix``, ``update``, ``close``) and falls back to the regular
``tqdm.auto.tqdm`` outside notebook environments so terminal / CI runs
still see standard one-line bars.

The public surface is intentionally tiny: ``get_tqdm()`` returns the
class to instantiate. Modules that need a progress bar import it at
module top so callers can monkey-patch a single attribute if they want
a different bar (e.g. set ``src.train.tqdm = my_class``), but the
default behavior is correct without any patching.
"""

from __future__ import annotations

import html as _html
import sys
import time
from typing import Any


def _is_in_colab() -> bool:
    """Heuristic: are we running inside a Colab notebook kernel?

    We avoid importing ``google.colab`` directly because that import has
    side effects on machines where it does exist (auth widgets, etc.).
    The two markers below are stable enough across Colab kernel versions
    to use for a UI choice.
    """
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in __import__("os").environ


class ColabVisibleTqdm:
    """Plain-HTML tqdm-shaped progress bar that renders reliably in Colab.

    Implements the subset of the tqdm API used by the trainer + LoRA
    loop. Refreshes are throttled to ``mininterval`` seconds (default
    0.25s) so per-batch update calls do not flood the notebook with
    HTML updates.
    """

    def __init__(
        self,
        iterable=None,
        total=None,
        desc=None,
        unit="it",
        leave=True,
        disable=False,
        mininterval=0.25,
        dynamic_ncols=True,
        **kwargs: Any,
    ):
        self.iterable = iterable
        self.desc = str(desc or "")
        self.unit = unit or "it"
        self.leave = True
        self.disable = bool(disable)
        self.mininterval = float(mininterval)
        self.n = 0
        self.postfix: dict = {}
        self.start_time = time.time()
        self.last_render = 0.0

        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except Exception:
                total = None
        self.total = total

        self._handle = None
        if not self.disable:
            try:
                from IPython.display import HTML, display

                self._handle = display(HTML(self._render_html()), display_id=True)
            except Exception:
                self._handle = None

    def __iter__(self):
        if self.iterable is None:
            return
        for item in self.iterable:
            self.n += 1
            self.refresh()
            yield item
        self.refresh(force=True)

    def update(self, n: int = 1) -> None:
        self.n += int(n)
        self.refresh(force=(n == 0))

    def set_description(self, desc=None, refresh: bool = True) -> None:
        self.desc = str(desc or "")
        if refresh:
            self.refresh(force=True)

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs: Any) -> None:
        data: dict = {}
        if ordered_dict:
            data.update(dict(ordered_dict))
        data.update(kwargs)
        self.postfix = data
        if refresh:
            self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        if self.disable or self._handle is None:
            return

        now = time.time()
        if not force and (now - self.last_render) < self.mininterval:
            return

        self.last_render = now
        try:
            from IPython.display import HTML

            self._handle.update(HTML(self._render_html()))
        except Exception:
            pass

    def close(self) -> None:
        self.refresh(force=True)

    def __enter__(self) -> "ColabVisibleTqdm":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def _render_html(self) -> str:
        elapsed = max(time.time() - self.start_time, 1e-9)
        rate = self.n / elapsed

        desc = _html.escape(self.desc)

        if self.total:
            pct = min(100.0, 100.0 * self.n / max(1, self.total))
            counter = f"{self.n:,}/{self.total:,}"
            progress = (
                f"<progress value='{self.n}' max='{self.total}' "
                f"style='width:460px;height:18px;vertical-align:middle;'></progress> "
                f"{pct:5.1f}%"
            )
        else:
            counter = f"{self.n:,}"
            progress = f"<span>{counter}</span>"

        postfix = ""
        if self.postfix:
            postfix = " | " + " | ".join(
                f"{_html.escape(str(k))}={_html.escape(str(v))}"
                for k, v in self.postfix.items()
            )

        return f"""
        <div style="
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 13px;
            line-height: 1.55;
            margin: 6px 0;
            white-space: nowrap;
        ">
            <b>{desc}</b><br>
            {progress}
            &nbsp; {counter} {self.unit}
            &nbsp; {rate:,.1f} {self.unit}/s
            {postfix}
        </div>
        """


def get_tqdm():
    """Return the best progress-bar class for the current environment.

    Inside Colab, ``ColabVisibleTqdm`` is preferred because the default
    widget-based ``tqdm.notebook`` silently fails for nested bars.
    Everywhere else we hand back ``tqdm.auto.tqdm`` so terminal / CI
    runs see normal one-line progress bars.
    """
    if _is_in_colab():
        return ColabVisibleTqdm
    try:
        from tqdm.auto import tqdm as _tqdm  # type: ignore

        return _tqdm
    except Exception:  # pragma: no cover - tqdm is optional
        def _noop_tqdm(x, *args, **kwargs):
            return x

        return _noop_tqdm


__all__ = ["ColabVisibleTqdm", "get_tqdm"]
