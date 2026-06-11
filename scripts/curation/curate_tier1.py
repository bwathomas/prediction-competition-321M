#!/usr/bin/env python
"""Curate external-benchmark training data: eval-arena, SWE-bench experiments, HELM.

Outputs /tmp/curation/<source>.parquet + /tmp/curation/report.json.
"""
import json
import os
import re
import sys
import threading
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yaml
from rapidfuzz import fuzz

OUT = "/tmp/curation"
BENCHDATA = "/tmp/benchdata/data"

# ---------------------------------------------------------------- matcher

SEP_RE = re.compile(r"[\s\-_./\\()+:,]+")
DATE_SUFFIX_RE = re.compile(r"(?: (?:20\d{6}|20\d{2} \d{2} \d{2}|\d{4}))+$")
# NOTE: deliberately excludes 'chat'/'instruct' — empirically those merges
# conflate base models with chat/instruct variants (different subjects).
DROP_TOKENS = {"preview", "latest", "hf"}
ORG_TOKENS = {
    "openai", "anthropic", "google", "googleai", "deepmind", "meta", "facebook",
    "mistralai", "cohere", "cohereforai", "qwen", "alibaba", "deepseek",
    "microsoft", "bigcode", "salesforce", "thudm", "databricks", "huggingfaceh4",
    "allenai", "ai21", "tiiuae", "nvidia", "xai", "amazon", "writer",
    "snowflake", "stabilityai", "upstage", "lmsys", "ibm", "naver", "adept",
    "aisingapore", "damo", "sambanova", "codellama", "wizardlm", "moonshotai",
    "zhipuai", "baichuan", "intern", "internlm", "openbmb", "togethercomputer",
}
ORG_TOKENS2 = {("deepseek", "ai"), ("01", "ai"),
               ("nous", "research"), ("zhipu", "ai")}


def norm(s):
    return SEP_RE.sub(" ", s.lower().strip()).strip()


def primary_variants(s):
    """Ordered list of normalized variants (full form first, then provider-stripped)."""
    out = []

    def add(v):
        if v and v not in out:
            out.append(v)

    n0 = norm(s)
    add(n0)
    if "/" in s:
        add(norm(s.split("/")[-1]))
    if "--" in s:
        add(norm(s.split("--", 1)[1]))
    toks = n0.split()
    if len(toks) > 2 and (toks[0], toks[1]) in ORG_TOKENS2:
        add(" ".join(toks[2:]))
    if len(toks) > 1 and toks[0] in ORG_TOKENS:
        add(" ".join(toks[1:]))
    return out


def fallback_variants(pvars):
    """Date-stripped / drop-token variants derived from primary variants."""
    out = []

    def add(v):
        if v and v not in pvars and v not in out:
            out.append(v)

    for v in pvars:
        nd = DATE_SUFFIX_RE.sub("", v).strip()
        add(nd)
        toks = [t for t in v.split() if t not in DROP_TOKENS]
        add(" ".join(toks))
        toks2 = [t for t in nd.split() if t not in DROP_TOKENS]
        add(" ".join(toks2))
    return out


def digit_tokens(s):
    return sorted(t for t in s.split() if any(c.isdigit() for c in t))


def equiv_pick(hits, raw):
    """Resolve a multi-hit set. If all hits are norm-equivalent duplicates of the
    same underlying model (their primary-variant sets share a common form), pick
    the display closest to the raw source string (deterministic). Else None."""
    hits = sorted(hits)
    if len(hits) == 1:
        return hits[0]
    common = set(primary_variants(hits[0]))
    for d in hits[1:]:
        common &= set(primary_variants(d))
        if not common:
            return None
    rl = raw.lower()
    return max(hits, key=lambda d: (fuzz.ratio(rl, d.lower()), d))


class Matcher:
    def __init__(self, subjects_df):
        self.exact_map = defaultdict(set)   # variant of display_name -> displays
        self.alias_map = defaultdict(set)   # variant of alias -> displays
        self.fb_map = defaultdict(set)      # fallback variant of display/alias -> displays
        self.fuzzy_cands = []               # (variant_string, display)
        self._cache = {}
        seen_fuzzy = set()
        for _, row in subjects_df.iterrows():
            disp = row["display_name"]
            aliases = list(row["raw_labels_seen"]) if row["raw_labels_seen"] is not None else []
            dvars = primary_variants(disp)
            for v in dvars:
                self.exact_map[v].add(disp)
            for v in fallback_variants(dvars):
                self.fb_map[v].add(disp)
            all_vars = list(dvars)
            for a in aliases:
                avars = primary_variants(a)
                for v in avars:
                    self.alias_map[v].add(disp)
                for v in fallback_variants(avars):
                    self.fb_map[v].add(disp)
                all_vars.extend(avars)
            for v in all_vars:
                if (v, disp) not in seen_fuzzy:
                    seen_fuzzy.add((v, disp))
                    self.fuzzy_cands.append((v, disp))

    def match(self, name, extra=()):
        key = (name, tuple(extra))
        if key in self._cache:
            return self._cache[key]
        res = self._match(name, extra)
        self._cache[key] = res
        return res

    def _match(self, name, extra):
        qs = primary_variants(name)
        for e in extra:
            for v in primary_variants(e):
                if v not in qs:
                    qs.append(v)
        # tier 1: exact-normalized against display names
        for q in qs:
            hit = self.exact_map.get(q)
            if hit:
                pick = equiv_pick(hit, name)
                if pick is not None:
                    return pick, "exact"
        # tier 2: alias-normalized
        for q in qs:
            hit = self.alias_map.get(q)
            if hit:
                pick = equiv_pick(hit, name)
                if pick is not None:
                    return pick, "alias"
        # tier 3: fallback normalization (date suffix / drop tokens); hits must
        # all be norm-equivalent duplicates of one model, else rejected
        fqs = fallback_variants(qs)
        for q in fqs + qs:
            hits = set()
            for m in (self.exact_map, self.alias_map, self.fb_map):
                hits |= m.get(q, set())
            if hits:
                pick = equiv_pick(hits, name)
                if pick is not None:
                    return pick, "fallback"
        # tier 4: high-confidence fuzzy (also over fallback-normalized queries;
        # guarded by dual ratio >= 95 + identical digit-token multisets)
        best_score, best_disps = 0, set()
        for q in qs + fqs:
            # if q had dict hits, they failed the equivalence check above
            # (ambiguous siblings, e.g. dated snapshots) — fuzzy must not
            # re-resolve that ambiguity via the digit guard
            if any(q in m for m in (self.exact_map, self.alias_map, self.fb_map)):
                continue
            qd = digit_tokens(q)
            for cand, disp in self.fuzzy_cands:
                s1 = fuzz.token_set_ratio(q, cand)
                if s1 < 95:
                    continue
                s2 = fuzz.token_sort_ratio(q, cand)
                sc = min(s1, s2)
                if sc < 95:
                    continue
                if digit_tokens(cand) != qd:
                    continue
                if sc > best_score:
                    best_score, best_disps = sc, {disp}
                elif sc == best_score:
                    best_disps.add(disp)
        if best_disps:
            pick = equiv_pick(best_disps, name)
            if pick is not None:
                return pick, "fuzzy"
        return None, None


# ---------------------------------------------------------------- helpers

_tls = threading.local()


def _session():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers["User-Agent"] = "bench-curation/0.1"
    return _tls.s


def http_get(url, timeout=60, as_json=False):
    r = _session().get(url, timeout=timeout)
    r.raise_for_status()
    return r.json() if as_json else r.text


def finalize(rows):
    df = pd.DataFrame(rows, columns=["subject_match", "source_model", "benchmark",
                                     "condition", "item_id", "item_text",
                                     "label", "match_tier"])
    df["label"] = df["label"].astype(float).clip(0.0, 1.0)
    return df


def source_report(df, models_seen, unmatched_rowcounts, failures):
    return {
        "n_rows": int(len(df)),
        "n_models_source": len(models_seen),
        "n_matched_subjects": int(df["subject_match"].nunique()) if len(df) else 0,
        "match_tier_counts": df["match_tier"].value_counts().to_dict() if len(df) else {},
        "n_unmatched_models": len(unmatched_rowcounts),
        "top_unmatched_models": dict(Counter(unmatched_rowcounts).most_common(20)),
        "failures": failures,
    }


# ---------------------------------------------------------------- source 1: eval-arena

EVAL_ARENA_FILES = {
    "humaneval+.jsonl": ("evalplus_humaneval", "none"),
    "mbpp+.jsonl": ("evalplus_mbpp", "none"),
    "ds1000.jsonl": ("ds1000", "none"),
    "swebench-verified.jsonl": ("swebench_verified_evalarena", "none"),
    "cruxeval_input_T0.2.jsonl": ("cruxeval", "input_T0.2"),
    "cruxeval_input_T0.8.jsonl": ("cruxeval", "input_T0.8"),
    "cruxeval_output_T0.2.jsonl": ("cruxeval", "output_T0.2"),
    "cruxeval_output_T0.8.jsonl": ("cruxeval", "output_T0.8"),
}


def run_eval_arena(matcher):
    rows, unmatched, failures = [], Counter(), []
    models_seen = set()
    for fname, (bench, cond0) in EVAL_ARENA_FILES.items():
        path = os.path.join(BENCHDATA, fname)
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    raw_model = d["model"]
                    models_seen.add(raw_model)
                    cond = cond0
                    match_name = raw_model
                    if bench == "cruxeval" and raw_model.endswith("+cot"):
                        match_name = raw_model[:-4]
                        cond = cond0 + "+cot"
                    disp, tier = matcher.match(match_name)
                    if disp is None:
                        unmatched[raw_model] += 1
                        continue
                    rows.append((disp, raw_model, bench, cond, d["example_id"],
                                 "", float(d["pass1"]), tier))
        except Exception as e:
            failures.append(f"{fname}: {type(e).__name__}: {e}")
    df = finalize(rows)
    df.to_parquet(os.path.join(OUT, "eval_arena.parquet"), index=False)
    return df, source_report(df, models_seen, unmatched, failures)


# ---------------------------------------------------------------- source 2: SWE-bench experiments

RAW_BASE = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation"
NEG_KEYS = ("unresolved", "failed", "error")


def _try_get(urls):
    last = None
    for u in urls:
        try:
            return http_get(u)
        except Exception as e:
            last = e
    raise last


def _fetch_submission(split, dirname):
    base = f"{RAW_BASE}/{split}/{dirname}"
    meta = yaml.safe_load(_try_get([f"{base}/metadata.yaml", f"{base}/metadata.yml"]))
    labeled = []
    try:
        results = json.loads(http_get(f"{base}/results/results.json"))
        for iid in results.get("resolved", []) or []:
            labeled.append((iid, 1.0))
        for k in NEG_KEYS:
            v = results.get(k)
            if isinstance(v, list):
                labeled.extend((iid, 0.0) for iid in v)
    except Exception:
        # bash-only style: per_instance_details.json {iid: {resolved: bool}}
        details = json.loads(http_get(f"{base}/per_instance_details.json"))
        for iid, d in details.items():
            if isinstance(d, dict) and "resolved" in d:
                labeled.append((iid, 1.0 if d["resolved"] else 0.0))
    return meta, labeled


def _model_from_meta(meta, dirname):
    tags = meta.get("tags") if isinstance(meta, dict) else None
    if isinstance(tags, dict):
        m = tags.get("model")
        if isinstance(m, list):
            m = [str(x) for x in m if x]
            if len(m) == 1:
                return m[0]
            if m:
                return " + ".join(m)
        elif isinstance(m, str) and m:
            return m
    info = meta.get("info") if isinstance(meta, dict) else None
    if isinstance(info, dict) and info.get("name"):
        return str(info["name"])
    return dirname


def run_swebench(matcher):
    rows, unmatched, failures = [], Counter(), []
    models_seen = set()
    splits = {"bash-only": ("swebench_bashonly", "/tmp/curation/gh_bashonly.json"),
              "verified": ("swebench_verified", "/tmp/curation/gh_verified.json")}
    jobs = []
    for split, (bench, listing) in splits.items():
        try:
            data = json.load(open(listing))
            if isinstance(data, dict):
                raise RuntimeError(f"GitHub API error: {data.get('message')}")
            dirs = [x["name"] for x in data if x["type"] == "dir"]
        except Exception as e:
            failures.append(f"listing {split}: {type(e).__name__}: {e}")
            continue
        jobs.extend((split, bench, d) for d in dirs)

    def work(job):
        split, bench, dirname = job
        meta, labeled = _fetch_submission(split, dirname)
        return job, meta, labeled

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for fut in as_completed(futs):
            split, bench, dirname = futs[fut]
            try:
                _, meta, labeled = fut.result()
            except Exception as e:
                failures.append(f"{split}/{dirname}: {type(e).__name__}: {e}")
                continue
            model = _model_from_meta(meta, dirname)
            models_seen.add(model)
            disp, tier = matcher.match(model, extra=(dirname,))
            if disp is None:
                unmatched[model] += len(labeled)
                continue
            for iid, lab in labeled:
                rows.append((disp, model, bench, dirname, str(iid), "", lab, tier))
    df = finalize(rows)
    df.to_parquet(os.path.join(OUT, "swebench_experiments.parquet"), index=False)
    return df, source_report(df, models_seen, unmatched, failures)


# ---------------------------------------------------------------- source 3: HELM

HELM_BASE = "https://storage.googleapis.com/crfm-helm-public"
HELM_SUITES = {
    "thaiexam": ("helm_thaiexam", "/tmp/curation/thaiexam_runs.json"),
    "mmlu-winogrande-afr": ("helm_afr", "/tmp/curation/afr_runs.json"),
}
MAX_ITEM_TEXT = 4000


def parse_run_name(run_name):
    scenario, _, argstr = run_name.partition(":")
    model, other = None, []
    for part in argstr.split(","):
        if part.startswith("model="):
            model = part[len("model="):]
        elif part:
            other.append(part)
    scen_spec = scenario + (":" + ",".join(other) if other else "")
    return scenario, model, scen_spec


def run_helm(matcher):
    rows, unmatched, failures = [], Counter(), []
    models_seen = set()
    inst_cache = {}
    inst_lock = threading.Lock()

    def get_instances(suite, run_suite, scen_spec, run_name):
        key = (suite, run_suite, scen_spec)
        with inst_lock:
            if key in inst_cache:
                return inst_cache[key]
        url = f"{HELM_BASE}/{suite}/benchmark_output/runs/{run_suite}/{run_name}/instances.json"
        try:
            instances = json.loads(http_get(url))
            texts = {}
            for inst in instances:
                iid = inst.get("id")
                txt = (inst.get("input") or {}).get("text") or ""
                if iid is not None:
                    texts[iid] = txt[:MAX_ITEM_TEXT]
        except Exception as e:
            failures.append(f"instances {suite}/{run_name}: {type(e).__name__}: {e}")
            texts = {}
        with inst_lock:
            inst_cache[key] = texts
        return texts

    def work(suite, bench, run_name, run_suite):
        scenario, model, scen_spec = parse_run_name(run_name)
        if model is None:
            return None
        url = f"{HELM_BASE}/{suite}/benchmark_output/runs/{run_suite}/{run_name}/display_predictions.json"
        preds = json.loads(http_get(url))
        texts = get_instances(suite, run_suite, scen_spec, run_name)
        # average over trials per instance
        agg = defaultdict(list)
        for p in preds:
            stats = p.get("stats") or {}
            val = stats.get("exact_match", stats.get("quasi_exact_match"))
            if val is None:
                continue
            agg[p["instance_id"]].append(float(val))
        out = []
        for iid, vals in agg.items():
            out.append((model, bench, scen_spec, f"{scen_spec}/{iid}",
                        texts.get(iid, ""), sum(vals) / len(vals)))
        return out

    jobs = []
    for suite, (bench, runs_file) in HELM_SUITES.items():
        try:
            runs = json.load(open(runs_file))
        except Exception as e:
            failures.append(f"runs listing {suite}: {type(e).__name__}: {e}")
            continue
        jobs.extend((suite, bench, rn, rs) for rn, rs in runs.items())

    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(work, *j): j for j in jobs}
        for fut in as_completed(futs):
            suite, bench, run_name, run_suite = futs[fut]
            done += 1
            if done % 100 == 0:
                print(f"  helm: {done}/{len(jobs)} runs", flush=True)
            try:
                out = fut.result()
            except Exception as e:
                failures.append(f"run {suite}/{run_name}: {type(e).__name__}: {e}")
                continue
            if not out:
                continue
            model = out[0][0]
            models_seen.add(model)
            alt = model.split("_", 1)[1] if "_" in model else model
            disp, tier = matcher.match(model, extra=(alt,))
            if disp is None:
                unmatched[model] += len(out)
                continue
            for model_, bench_, cond, iid, text, lab in out:
                rows.append((disp, model_, bench_, cond, iid, text, lab, tier))
    df = finalize(rows)
    df.to_parquet(os.path.join(OUT, "helm.parquet"), index=False)
    return df, source_report(df, models_seen, unmatched, failures)


# ---------------------------------------------------------------- main

def main():
    subjects = pd.read_parquet("/tmp/subjects.parquet")
    matcher = Matcher(subjects)
    print(f"matcher: {len(matcher.exact_map)} exact keys, {len(matcher.alias_map)} alias keys, "
          f"{len(matcher.fb_map)} fallback keys, {len(matcher.fuzzy_cands)} fuzzy candidates", flush=True)

    report = {"sources": {}, "overall": {}}
    rpath = os.path.join(OUT, "report.json")
    if os.path.exists(rpath):
        report = json.load(open(rpath))
    sources = [("eval_arena", run_eval_arena),
               ("swebench_experiments", run_swebench),
               ("helm", run_helm)]
    only = set(sys.argv[1:])
    if only:
        sources = [(n, f) for n, f in sources if n in only]
    total_rows, all_subjects = 0, set()
    for name, fn in sources:
        print(f"== {name}", flush=True)
        try:
            df, rep = fn(matcher)
            report["sources"][name] = rep
            total_rows += len(df)
            all_subjects |= set(df["subject_match"].unique())
            print(f"   {len(df)} rows, {rep['n_matched_subjects']} subjects matched, "
                  f"tiers={rep['match_tier_counts']}", flush=True)
        except Exception as e:
            traceback.print_exc()
            report["sources"][name] = {"fatal_error": f"{type(e).__name__}: {e}"}
    # recompute overall from all parquets on disk
    total_rows, all_subjects = 0, set()
    pq_files = sorted(f for f in os.listdir(OUT) if f.endswith(".parquet"))
    for f in pq_files:
        d = pd.read_parquet(os.path.join(OUT, f), columns=["subject_match"])
        total_rows += len(d)
        all_subjects |= set(d["subject_match"].unique())
    report["overall"] = {
        "total_rows": total_rows,
        "total_matched_subjects": len(all_subjects),
        "parquet_files": pq_files,
    }
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
