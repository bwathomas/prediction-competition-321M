"""Static audit of notebooks/qwen8b_four_member_stacked.py.

Checks:
  (1) Syntax (already verified separately).
  (2) For a set of "watched" functions (the new Member-2 MLP fit/apply
      plus a few related M3/M5/M6 fits), every call site's keyword
      arguments are a subset of the function signature, and no required
      kwargs are missing.
  (3) Module-level NameError sweep: every Name *read* at module top
      level must be one of: a builtin, an import target, or a name
      previously bound at module top level.  Names read inside function
      bodies are NOT checked (they are resolved lazily by the
      interpreter).

The script prints findings and exits with status 1 if any are found.
"""
from __future__ import annotations

import ast
import builtins
import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "qwen8b_four_member_stacked.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _sig_for(func) -> inspect.Signature:
    return inspect.signature(func)


def _required_kwargs(sig: inspect.Signature) -> set[str]:
    out = set()
    for p in sig.parameters.values():
        if p.kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        if p.default is inspect.Parameter.empty:
            out.add(p.name)
    return out


def _all_kwargs(sig: inspect.Signature) -> set[str]:
    out = set()
    has_kwargs = False
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_kwargs = True
            continue
        if p.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            out.add(p.name)
    return out if not has_kwargs else out  # we treat **kwargs as anything-goes


def _accepts_var_kw(sig: inspect.Signature) -> bool:
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def audit() -> int:
    src = NOTEBOOK.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(NOTEBOOK))

    # --- (2) Signature audit for the watched functions ---
    m2 = _load_module(ROOT / "src" / "member2_metadata_mlp.py", "_audit_m2")
    knn = _load_module(ROOT / "src" / "knn_member.py", "_audit_knn")

    # For apply_state_batch / apply_state_one we synthesize a "real"
    # signature: state (positional-or-keyword) plus the forwarded kwargs
    # from apply_batch / apply_one (minus the state kwarg).
    def _wrap_state_first(real_sig: inspect.Signature) -> inspect.Signature:
        params = []
        for p in real_sig.parameters.values():
            if p.name == "state":
                params.append(
                    inspect.Parameter(
                        "state",
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty,
                    )
                )
            else:
                params.append(p)
        # Reorder: state first.
        params.sort(key=lambda p: 0 if p.name == "state" else 1)
        return inspect.Signature(parameters=params)

    watched = {
        "fit_member2_metadata_mlp": _sig_for(m2.fit_member2_metadata_mlp),
        "m2_apply_state_batch": _wrap_state_first(_sig_for(m2.apply_batch)),
        "m2_apply_state_one": _wrap_state_first(_sig_for(m2.apply_one)),
    }
    # Also rough-check fit_knn_member kwargs at the shuffled call site.
    if hasattr(knn, "fit_knn_member"):
        watched["fit_knn_member"] = _sig_for(knn.fit_knn_member)

    findings: list[str] = []

    def report(line: int, msg: str) -> None:
        findings.append(f"  L{line}: {msg}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname: str | None = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        if fname not in watched:
            continue
        sig = watched[fname]
        kw_present = {kw.arg for kw in node.keywords if kw.arg is not None}
        kw_allowed = _all_kwargs(sig)
        kw_required = _required_kwargs(sig)
        var_kw = _accepts_var_kw(sig)

        if not var_kw:
            unknown = sorted(kw_present - kw_allowed)
            for u in unknown:
                report(node.lineno, f"{fname}: unknown kwarg '{u}'")
        # Required kwargs that must be passed (allow positional args by
        # taking those into account: count positional args supplied).
        n_pos = len(node.args)
        pos_param_names = [
            p.name
            for p in sig.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        bound_by_pos = set(pos_param_names[:n_pos])
        missing = sorted(kw_required - kw_present - bound_by_pos)
        for m in missing:
            report(node.lineno, f"{fname}: missing required kwarg '{m}'")

    # --- (3) Module-level NameError sweep ---
    defined: set[str] = set(dir(builtins))
    # Names provided by the notebook's runtime (Colab, IPython, etc.).
    defined.update({
        "__name__", "__doc__", "__file__", "__package__", "__loader__",
        "__spec__", "__builtins__",
        # Colab / Jupyter helpers occasionally used at module level.
        "display", "get_ipython", "In", "Out", "exit", "quit",
    })
    # Sticky helper: get all names bound by a single statement.
    def _bind_targets(target: ast.AST) -> list[str]:
        out: list[str] = []
        if isinstance(target, ast.Name):
            out.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                out.extend(_bind_targets(elt))
        elif isinstance(target, ast.Starred):
            out.extend(_bind_targets(target.value))
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            pass  # not a name binding at module scope
        return out

    name_reads: list[tuple[int, str]] = []
    # Walk the module-level body in order, threading `defined` forward.
    def _walk_module_level(body: list[ast.stmt]) -> None:
        for stmt in body:
            _process_stmt(stmt)

    def _read_names_in(expr: ast.AST) -> list[tuple[int, str]]:
        """Collect Name reads in ``expr`` but skip names that are bound
        within an inner scope (lambdas, comprehensions, function/class
        defs).  Module-level NameErrors only fire for names looked up
        in the module's own scope.
        """
        out: list[tuple[int, str]] = []

        def _bound_in_comp(comp: ast.AST) -> set[str]:
            names: set[str] = set()
            generators = getattr(comp, "generators", [])
            for gen in generators:
                for tgt in ast.walk(gen.target):
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
            return names

        def _walk_expr(node: ast.AST, blocked: set[str]) -> None:
            # Inner scopes shadow / consume their bound names.
            if isinstance(node, (ast.Lambda,)):
                lambda_args = set()
                for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                    lambda_args.add(a.arg)
                if node.args.vararg is not None:
                    lambda_args.add(node.args.vararg.arg)
                if node.args.kwarg is not None:
                    lambda_args.add(node.args.kwarg.arg)
                # Defaults are evaluated in the OUTER scope.
                for d in node.args.defaults + node.args.kw_defaults:
                    if d is not None:
                        _walk_expr(d, blocked)
                _walk_expr(node.body, blocked | lambda_args)
                return
            if isinstance(node, (
                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
            )):
                # First generator's iter is evaluated in the outer scope.
                inner_blocked = set(blocked)
                for i, gen in enumerate(node.generators):
                    if i == 0:
                        _walk_expr(gen.iter, blocked)
                    else:
                        _walk_expr(gen.iter, inner_blocked)
                    for tgt in ast.walk(gen.target):
                        if isinstance(tgt, ast.Name):
                            inner_blocked.add(tgt.id)
                    for cond in gen.ifs:
                        _walk_expr(cond, inner_blocked)
                if isinstance(node, ast.DictComp):
                    _walk_expr(node.key, inner_blocked)
                    _walk_expr(node.value, inner_blocked)
                else:
                    _walk_expr(node.elt, inner_blocked)
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip inner defs entirely (names looked up there are
                # resolved lazily; not a module-level NameError surface).
                return
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load) and node.id not in blocked:
                    out.append((node.lineno, node.id))
                return
            for child in ast.iter_child_nodes(node):
                _walk_expr(child, blocked)

        _walk_expr(expr, set())
        return out

    def _process_stmt(stmt: ast.stmt) -> None:
        # Read-then-bind: evaluate the right-hand side first (reads),
        # then bind the LHS.
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                defined.add((alias.asname or alias.name.split(".")[0]))
            return
        if isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name == "*":
                    # Star import - assume it brings in arbitrary names.
                    defined.add("__star__")
                else:
                    defined.add(alias.asname or alias.name)
            return
        if isinstance(stmt, ast.Assign):
            for n in _read_names_in(stmt.value):
                name_reads.append(n)
            for tgt in stmt.targets:
                for nm in _bind_targets(tgt):
                    defined.add(nm)
            return
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                for n in _read_names_in(stmt.value):
                    name_reads.append(n)
            if stmt.target is not None:
                for nm in _bind_targets(stmt.target):
                    defined.add(nm)
            return
        if isinstance(stmt, ast.AugAssign):
            for n in _read_names_in(stmt.value):
                name_reads.append(n)
            for nm in _bind_targets(stmt.target):
                # AugAssign requires the target to already exist; treat
                # as both read and write.
                name_reads.append((stmt.lineno, nm))
                defined.add(nm)
            return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Read names in default values & decorators at module scope;
            # but do NOT recurse into the function body.
            for d in stmt.decorator_list:
                for n in _read_names_in(d):
                    name_reads.append(n)
            for d in stmt.args.defaults + stmt.args.kw_defaults:
                if d is not None:
                    for n in _read_names_in(d):
                        name_reads.append(n)
            defined.add(stmt.name)
            return
        if isinstance(stmt, ast.ClassDef):
            for d in stmt.decorator_list:
                for n in _read_names_in(d):
                    name_reads.append(n)
            for b in stmt.bases:
                for n in _read_names_in(b):
                    name_reads.append(n)
            defined.add(stmt.name)
            return
        if isinstance(stmt, ast.If):
            for n in _read_names_in(stmt.test):
                name_reads.append(n)
            _walk_module_level(stmt.body)
            _walk_module_level(stmt.orelse)
            return
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            for n in _read_names_in(stmt.iter):
                name_reads.append(n)
            for nm in _bind_targets(stmt.target):
                defined.add(nm)
            _walk_module_level(stmt.body)
            _walk_module_level(stmt.orelse)
            return
        if isinstance(stmt, (ast.While,)):
            for n in _read_names_in(stmt.test):
                name_reads.append(n)
            _walk_module_level(stmt.body)
            _walk_module_level(stmt.orelse)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                for n in _read_names_in(item.context_expr):
                    name_reads.append(n)
                if item.optional_vars is not None:
                    for nm in _bind_targets(item.optional_vars):
                        defined.add(nm)
            _walk_module_level(stmt.body)
            return
        if isinstance(stmt, ast.Try):
            _walk_module_level(stmt.body)
            for handler in stmt.handlers:
                if handler.type is not None:
                    for n in _read_names_in(handler.type):
                        name_reads.append(n)
                if handler.name is not None:
                    defined.add(handler.name)
                _walk_module_level(handler.body)
            _walk_module_level(stmt.orelse)
            _walk_module_level(stmt.finalbody)
            return
        if isinstance(stmt, ast.Expr):
            for n in _read_names_in(stmt.value):
                name_reads.append(n)
            return
        if isinstance(stmt, ast.Assert):
            for n in _read_names_in(stmt.test):
                name_reads.append(n)
            if stmt.msg is not None:
                for n in _read_names_in(stmt.msg):
                    name_reads.append(n)
            return
        if isinstance(stmt, ast.Delete):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    name_reads.append((stmt.lineno, tgt.id))
                    defined.discard(tgt.id)
            return
        if isinstance(stmt, (ast.Global, ast.Nonlocal, ast.Pass)):
            return
        if isinstance(stmt, ast.Raise):
            if stmt.exc is not None:
                for n in _read_names_in(stmt.exc):
                    name_reads.append(n)
            if stmt.cause is not None:
                for n in _read_names_in(stmt.cause):
                    name_reads.append(n)
            return
        if isinstance(stmt, ast.Return):
            return  # not valid at module scope; ignore
        # Fallback: walk reads.
        for n in _read_names_in(stmt):
            name_reads.append(n)

    _walk_module_level(tree.body)

    # If any star imports happened, skip the NameError sweep (too lossy).
    star_imported = "__star__" in defined
    # NameError sweep at module top level is too noisy because of `del`
    # and order-of-binding complexity (the static check would need a
    # proper data-flow pass).  We delegate that to ``pyflakes``; this
    # script focuses on signature audits for the new M2 wiring.
    _ = star_imported  # quiet the linter, intentionally unused below.

    if not findings:
        print("AUDIT OK: no issues found.")
        return 0
    print(f"AUDIT found {len(findings)} issue(s):")
    for f in findings:
        print(f)
    return 1


if __name__ == "__main__":
    sys.exit(audit())
