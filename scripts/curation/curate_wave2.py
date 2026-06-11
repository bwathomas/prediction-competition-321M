#!/usr/bin/env python
"""Wave-2 curation: livebench, alpacaeval, taubench, arcagi, biggen, terminal_bench.

Usage: curate2.py <source>   (one source per process; writes <name>.parquet + r2_<name>.json)
       curate2.py merge      (assembles /tmp/curation/report2.json)

Reuses the audited matcher from curate.py unchanged.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import defusedxml.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yaml

sys.path.insert(0, "/tmp/curation")
from curate import Matcher, finalize, source_report, http_get  # noqa: E402

OUT = "/tmp/curation"
MAX_ITEM_TEXT = 4000
HF = "https://huggingface.co"

_tls = threading.local()


def sess():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers["User-Agent"] = "bench-curation/0.1"
    return _tls.s


def get_json(url, timeout=120, retries=3):
    last = None
    for i in range(retries):
        try:
            r = sess().get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


def get_parquet(url):
    r = sess().get(url, timeout=300)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def write_source(name, df, rep):
    df.to_parquet(os.path.join(OUT, f"{name}.parquet"), index=False)
    with open(os.path.join(OUT, f"r2_{name}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[{name}] {len(df)} rows, {rep.get('n_matched_subjects')} subjects, "
          f"tiers={rep.get('match_tier_counts')}", flush=True)


def ihash(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


# ---------------------------------------------------------------- livebench

LIVEBENCH_CATS = ["coding", "data_analysis", "instruction_following",
                  "language", "math", "reasoning"]


def run_livebench(matcher):
    failures = []
    mj = get_parquet(f"{HF}/api/datasets/livebench/model_judgment/parquet/default/leaderboard/0.parquet")
    # question text maps (cheap: 6 small parquets)
    qtext = {}
    for cat in LIVEBENCH_CATS:
        try:
            urls = get_json(f"{HF}/api/datasets/livebench/{cat}/parquet")
            flat = [u for cfg in urls.values() for split in cfg.values() for u in split]
            for u in flat:
                qdf = get_parquet(u)
                for qid, turns in zip(qdf["question_id"], qdf["turns"]):
                    if turns is not None and len(turns):
                        qtext[qid] = str(turns[0])[:MAX_ITEM_TEXT]
        except Exception as e:
            failures.append(f"questions {cat}: {type(e).__name__}: {e}")
    # average score per (model, question) over turns/dupes
    g = mj.groupby(["model", "category", "question_id"], as_index=False)["score"].mean()
    rows, unmatched = [], Counter()
    models_seen = set(g["model"].unique())
    for model, sub in g.groupby("model"):
        disp, tier = matcher.match(model)
        if disp is None:
            unmatched[model] += len(sub)
            continue
        for cat, qid, score in zip(sub["category"], sub["question_id"], sub["score"]):
            rows.append((disp, model, f"livebench_{cat}", "none", str(qid),
                         qtext.get(qid, ""), float(score), tier))
    df = finalize(rows)
    rep = source_report(df, models_seen, unmatched, failures)
    rep["notes"] = "label = per-question judge/scorer score clipped to [0,1]; mean over duplicate (model,question) rows"
    write_source("livebench", df, rep)


# ---------------------------------------------------------------- alpacaeval

AE_RAW = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results"


def run_alpacaeval(matcher):
    failures = []
    listing = get_json("https://api.github.com/repos/tatsu-lab/alpaca_eval/contents/results", timeout=60)
    if isinstance(listing, dict):
        raise RuntimeError(f"GitHub API: {listing.get('message')}")
    models = [x["name"] for x in listing if x["type"] == "dir"]
    models_seen = set(models)
    rows, unmatched = [], Counter()
    lock = threading.Lock()
    pref_minmax = [float("inf"), float("-inf")]

    def work(model):
        url = f"{AE_RAW}/{model}/weighted_alpaca_eval_gpt4_turbo/annotations.json"
        r = sess().get(url, timeout=120)
        if r.status_code == 404:
            return model, None
        r.raise_for_status()
        return model, r.json()

    no_file = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(work, m): m for m in models}
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                model, annots = fut.result()
            except Exception as e:
                failures.append(f"{m}: {type(e).__name__}: {e}")
                continue
            if annots is None:
                no_file.append(model)
                continue
            disp, tier = matcher.match(model)
            if disp is None:
                with lock:
                    unmatched[model] += len(annots)
                continue
            batch = []
            for a in annots:
                pref = a.get("preference")
                instr = a.get("instruction") or ""
                if pref is None:
                    continue
                pref = float(pref)
                # weighted annotators emit preference in [1,2]: 1 + P(model output preferred)
                label = pref - 1.0 if pref > 1.0 or pref == 1.0 else pref
                batch.append((disp, model, "alpacaeval", "vs_gpt4_turbo_weighted",
                              ihash(instr), instr[:MAX_ITEM_TEXT], label, tier))
                with lock:
                    pref_minmax[0] = min(pref_minmax[0], pref)
                    pref_minmax[1] = max(pref_minmax[1], pref)
            with lock:
                rows.extend(batch)
    df = finalize(rows)
    rep = source_report(df, models_seen, unmatched, failures)
    rep["notes"] = (f"label = preference - 1 = P(model output preferred over gpt4_turbo reference), "
                    f"weighted_alpaca_eval_gpt4_turbo annotator; raw preference range seen "
                    f"[{pref_minmax[0]:.3f}, {pref_minmax[1]:.3f}]; "
                    f"{len(no_file)} model dirs lacked weighted annotations (skipped)")
    rep["n_models_without_weighted_annotations"] = len(no_file)
    write_source("alpacaeval", df, rep)


# ---------------------------------------------------------------- taubench

TAU_S3 = "https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com"
TAU_DOMAIN_RE = re.compile(r"_(airline|retail|telecom|banking_knowledge)_")
TAU_MAX_FILE = 60 * 1024 * 1024     # skip >60MB trajectory files (banking mostly)
TAU_BYTE_BUDGET = 3_200_000_000
TAU_MAX_SUBS = 45
TAU_DEADLINE = 14 * 60              # seconds from source start


def s3_list_all():
    keys, token = [], None
    while True:
        url = f"{TAU_S3}/?list-type=2&max-keys=1000"
        if token:
            url += f"&continuation-token={requests.utils.quote(token)}"
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        root = ET.fromstring(sess().get(url, timeout=60).text)
        for c in root.findall(ns + "Contents"):
            keys.append((c.find(ns + "Key").text, int(c.find(ns + "Size").text)))
        if root.find(ns + "IsTruncated").text != "true":
            return keys
        token = root.find(ns + "NextContinuationToken").text


def run_taubench(matcher):
    t0 = time.time()
    failures, rows, unmatched = [], [], Counter()
    keys = s3_list_all()
    subs = defaultdict(list)   # subdir -> [(key, size, domain)]
    for k, sz in keys:
        parts = k.split("/")
        if len(parts) == 4 and parts[0] == "submissions" and parts[2] == "trajectories" \
                and k.endswith(".json") and not parts[1].startswith("A_EXAMPLE"):
            m = TAU_DOMAIN_RE.search(parts[3])
            if m:
                subs[parts[1]].append((k, sz, m.group(1)))
    # prefer cheapest submissions first so the cap keeps the most complete set
    ordered = sorted(subs.items(), key=lambda kv: sum(s for _, s, _ in kv[1]))[:TAU_MAX_SUBS]
    models_seen = set(d for d, _ in ordered)
    skipped_big, bytes_used = [], [0]
    lock = threading.Lock()

    def sub_model(subdir):
        try:
            meta = get_json(f"{TAU_S3}/submissions/{subdir}/submission.json", timeout=30)
            return meta.get("model_name") or subdir.split("_")[0]
        except Exception:
            return subdir.split("_")[0]

    def work(subdir, key, size, domain):
        with lock:
            if bytes_used[0] + size > TAU_BYTE_BUDGET:
                return None
            bytes_used[0] += size
        d = get_json(f"{TAU_S3}/{key}", timeout=600, retries=3)
        tasks = {}
        for t in d.get("tasks", []):
            tid = str(t.get("id"))
            desc = t.get("description") or {}
            us = t.get("user_scenario") or {}
            txt = json.dumps({"purpose": (desc.get("purpose") if isinstance(desc, dict) else None),
                              "instructions": us.get("instructions")}, ensure_ascii=False)
            tasks[tid] = txt[:MAX_ITEM_TEXT]
        agg = defaultdict(list)
        for sim in d.get("simulations", []):
            ri = sim.get("reward_info") or {}
            rw = ri.get("reward")
            if rw is None:
                continue
            agg[str(sim.get("task_id"))].append(float(rw))
        return subdir, domain, tasks, agg

    jobs = []
    for subdir, files in ordered:
        for k, sz, dom in files:
            if sz > TAU_MAX_FILE:
                skipped_big.append((k, sz))
                continue
            jobs.append((subdir, k, sz, dom))
    model_cache = {}
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(work, *j): j for j in jobs}
        for fut in as_completed(futs):
            subdir, key, _, _ = futs[fut]
            done += 1
            if done % 10 == 0:
                print(f"  tau: {done}/{len(jobs)} files, {bytes_used[0]/1e9:.2f}GB, "
                      f"{time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > TAU_DEADLINE:
                failures.append("deadline hit: remaining files skipped")
                for f2 in futs:
                    f2.cancel()
                break
            try:
                res = fut.result()
            except Exception as e:
                failures.append(f"{key}: {type(e).__name__}: {e}")
                continue
            if res is None:
                failures.append(f"{key}: skipped, byte budget exhausted")
                continue
            subdir, domain, tasks, agg = res
            if subdir not in model_cache:
                model_cache[subdir] = sub_model(subdir)
            model = model_cache[subdir]
            disp, tier = matcher.match(model, extra=(subdir.split("_")[0],))
            if disp is None:
                unmatched[model] += len(agg)
                continue
            for tid, rewards in agg.items():
                rows.append((disp, model, f"taubench_{domain}",
                             f"trials{len(rewards)}", f"{domain}/{tid}",
                             tasks.get(tid, ""), sum(rewards) / len(rewards), tier))
    df = finalize(rows)
    rep = source_report(df, models_seen, unmatched, failures)
    rep["notes"] = (f"label = mean reward_info.reward over trials per (model,task); "
                    f"capped at {TAU_MAX_SUBS} cheapest submissions, files>{TAU_MAX_FILE>>20}MB skipped "
                    f"({len(skipped_big)} skipped, mostly banking_knowledge); "
                    f"{bytes_used[0]/1e9:.2f}GB downloaded")
    write_source("taubench", df, rep)


# ---------------------------------------------------------------- arcagi

ARC_REPOS = {
    "arcagi_v1": ("https://huggingface.co/datasets/arcprize/arc_agi_v1_public_eval",
                  "https://github.com/fchollet/ARC-AGI"),
    "arcagi_v2": ("https://huggingface.co/datasets/arcprize/arc_agi_v2_public_eval",
                  "https://github.com/arcprize/ARC-AGI-2"),
}
EFFORT_RE = re.compile(r"-(high|low|medium|minimal)$")
THINK_RE = re.compile(r"-thinking-(\d+[kK]|none)$")


def arc_parse(mdir):
    """Split an ARC model dir into (match_name, condition_prefix); thinking
    budget / reasoning effort are run conditions, not subject identity."""
    conds, name = [], mdir
    m = EFFORT_RE.search(name)
    if m:
        conds.append(m.group(1))
        name = name[:m.start()]
    m = THINK_RE.search(name)
    if m:
        conds.append(f"thinking_{m.group(1)}")
        name = name[:m.start()]
    elif name.endswith("-thinking"):
        conds.append("thinking")
        name = name[:-len("-thinking")]
    return name, ("|".join(reversed(conds)) + "|" if conds else "")


def clone(url, dest):
    if os.path.isdir(dest):
        return
    subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, dest],
                   check=True, timeout=600,
                   env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"})


def grid_eq(a, b):
    try:
        return [list(map(int, r)) for r in a] == [list(map(int, r)) for r in b]
    except Exception:
        return False


def run_arcagi(matcher):
    failures, rows, unmatched = [], [], Counter()
    models_seen = set()
    base = "/tmp/curation/_arc"
    os.makedirs(base, exist_ok=True)
    for bench, (eval_url, sol_url) in ARC_REPOS.items():
        try:
            eval_dir = os.path.join(base, bench + "_eval")
            sol_dir = os.path.join(base, bench + "_sol")
            clone(eval_url, eval_dir)
            clone(sol_url, sol_dir)
            sols, texts = {}, {}
            sdir = os.path.join(sol_dir, "data", "evaluation")
            for fn in os.listdir(sdir):
                if not fn.endswith(".json"):
                    continue
                tid = fn[:-5]
                task = json.load(open(os.path.join(sdir, fn)))
                sols[tid] = [t.get("output") for t in task.get("test", [])]
                texts[tid] = json.dumps(
                    {"train": task.get("train"),
                     "test_input": [t.get("input") for t in task.get("test", [])]},
                    separators=(",", ":"))[:MAX_ITEM_TEXT]
            for mdir in sorted(os.listdir(eval_dir)):
                mpath = os.path.join(eval_dir, mdir)
                if mdir.startswith(".") or not os.path.isdir(mpath):
                    continue
                models_seen.add(mdir)
                match_name, think = arc_parse(mdir)
                disp, tier = matcher.match(match_name)
                taskfiles = [f for f in os.listdir(mpath) if f.endswith(".json")]
                if disp is None:
                    unmatched[mdir] += len(taskfiles)
                    continue
                for fn in taskfiles:
                    tid = fn[:-5]
                    if tid not in sols:
                        continue
                    try:
                        attempts = json.load(open(os.path.join(mpath, fn)))
                    except Exception:
                        failures.append(f"{bench}/{mdir}/{fn}: unreadable")
                        continue
                    outs = sols[tid]
                    per_test, n_att = [], 0
                    for i, sol_out in enumerate(outs):
                        entry = attempts[i] if isinstance(attempts, list) and i < len(attempts) else {}
                        ok = False
                        if isinstance(entry, dict):
                            atts = [v for k, v in entry.items() if k.startswith("attempt")]
                            n_att = max(n_att, len(atts))
                            for av in atts:
                                g = av.get("answer") if isinstance(av, dict) else av
                                if g is not None and grid_eq(g, sol_out):
                                    ok = True
                                    break
                        per_test.append(1.0 if ok else 0.0)
                    if not per_test:
                        continue
                    rows.append((disp, mdir, bench, f"{think}attempts{n_att}",
                                 tid, texts.get(tid, ""),
                                 sum(per_test) / len(per_test), tier))
        except Exception as e:
            failures.append(f"{bench}: {type(e).__name__}: {e}")
            traceback.print_exc()
    df = finalize(rows)
    rep = source_report(df, models_seen, unmatched, failures)
    rep["notes"] = ("label = mean over test inputs of any-attempt exact-match; thinking budget "
                    "kept as condition (model matched on thinking-stripped name)")
    write_source("arcagi", df, rep)


# ---------------------------------------------------------------- biggen

def run_biggen(matcher):
    failures, rows, unmatched = [], [], Counter()
    urls = get_json(f"{HF}/api/datasets/prometheus-eval/BiGGen-Bench-Results/parquet")
    parts = urls["default"]["llm_as_a_judge"]
    dfs = []
    cached = "/tmp/curation/_biggen_part0.parquet"
    for i, u in enumerate(parts):
        if i == 0 and os.path.exists(cached):
            dfs.append(pd.read_parquet(cached))
        else:
            dfs.append(get_parquet(u))
    raw = pd.concat(dfs, ignore_index=True)
    models_seen = set(raw["model_name"].unique())

    def scalar(v):
        try:
            if v is None or hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                return None
            return float(v)
        except Exception:
            return None

    for model, sub in raw.groupby("model_name"):
        disp, tier = matcher.match(model)
        if disp is None:
            unmatched[model] += len(sub)
            continue
        for rid, cap, inp, sc in zip(sub["id"], sub["capability"], sub["input"], sub["gpt4_score"]):
            s = scalar(sc)
            if s is None or not (1.0 <= s <= 5.0):   # also rejects NaN
                continue
            rows.append((disp, model, f"biggen_{cap}", "judge_gpt4", str(rid),
                         (str(inp) if inp is not None else "")[:MAX_ITEM_TEXT],
                         (s - 1.0) / 4.0, tier))
    df = finalize(rows)
    rep = source_report(df, models_seen, unmatched, failures)
    rep["notes"] = "label = (gpt4_score - 1)/4 from llm_as_a_judge split (gpt4 judge has 100% coverage)"
    write_source("biggen", df, rep)


# ---------------------------------------------------------------- terminal_bench

TB1_GIT = "https://github.com/laude-institute/terminal-bench-leaderboard"
TB2_API = f"{HF}/api/datasets/harborframework/terminal-bench-2-leaderboard/tree/main"
TB2_RAW = f"{HF}/datasets/harborframework/terminal-bench-2-leaderboard/resolve/main"
TB2_ROOT = "submissions/terminal-bench/2.0"


def run_tb1(matcher, rows, unmatched, models_seen, failures):
    dest = "/tmp/curation/_tb1"
    clone(TB1_GIT, dest)
    resdir = os.path.join(dest, "results")
    for suite in os.listdir(resdir):
        spath = os.path.join(resdir, suite)
        if not os.path.isdir(spath):
            continue
        for run in os.listdir(spath):
            rpath = os.path.join(spath, run)
            if not os.path.isdir(rpath):
                continue
            parts = run.split("_")
            agent = parts[1] if len(parts) > 2 else run
            model = "_".join(parts[2:]) if len(parts) > 2 else run
            models_seen.add(model)
            agg = defaultdict(list)   # task_id -> [resolved...]
            instr = {}
            for dirpath, _, files in os.walk(rpath):
                if "results.json" not in files:
                    continue
                try:
                    d = json.load(open(os.path.join(dirpath, "results.json")))
                except Exception:
                    continue
                if not isinstance(d, dict) or "is_resolved" not in d:
                    continue
                tid = d.get("task_id") or os.path.basename(os.path.dirname(dirpath))
                agg[tid].append(1.0 if d["is_resolved"] else 0.0)
                if tid not in instr and d.get("instruction"):
                    instr[tid] = str(d["instruction"])[:MAX_ITEM_TEXT]
            if not agg:
                continue
            disp, tier = matcher.match(model)
            if disp is None:
                # retry on underscore-suffixes that still carry a digit (version),
                # e.g. 'agent_claude-4-5-sonnet' -> 'claude-4-5-sonnet'; bare family
                # words like 'sonnet' are excluded by the digit guard
                mparts = model.split("_")
                for i in range(1, len(mparts)):
                    ex = "_".join(mparts[i:])
                    if not any(c.isdigit() for c in ex):
                        continue
                    disp, tier = matcher.match(ex)
                    if disp is not None:
                        break
            if disp is None:
                unmatched[f"tb1:{model}"] += len(agg)
                continue
            for tid, vals in agg.items():
                rows.append((disp, model, "terminal_bench_1",
                             f"{agent}|trials{len(vals)}", str(tid),
                             instr.get(tid, ""), sum(vals) / len(vals), tier))


def run_tb2(matcher, rows, unmatched, models_seen, failures):
    subs = [x["path"].split("/")[-1] for x in get_json(f"{TB2_API}/{TB2_ROOT}", timeout=60)
            if x["type"] == "directory"]

    def work(sub):
        meta = yaml.safe_load(sess().get(f"{TB2_RAW}/{TB2_ROOT}/{sub}/metadata.yaml", timeout=60).text)
        entries = get_json(f"{TB2_API}/{TB2_ROOT}/{sub}", timeout=60)
        results = []
        for e in entries:
            if e["type"] != "directory":
                continue
            run = e["path"].split("/")[-1]
            r = sess().get(f"{TB2_RAW}/{TB2_ROOT}/{sub}/{run}/result.json", timeout=120)
            if r.status_code != 200:
                continue
            results.append((run, r.json()))
        return sub, meta, results

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(work, s): s for s in subs}
        for fut in as_completed(futs):
            sub = futs[fut]
            try:
                sub, meta, results = fut.result()
            except Exception as e:
                failures.append(f"tb2 {sub}: {type(e).__name__}: {e}")
                continue
            agent = sub.split("__")[0]
            model = sub.split("__")[-1]
            mdisp = None
            if isinstance(meta, dict):
                agent = meta.get("agent_display_name") or agent
                mods = meta.get("models") or []
                if mods and isinstance(mods[0], dict):
                    mdisp = mods[0].get("model_display_name") or mods[0].get("model_name")
            model_name = mdisp or model
            models_seen.add(model_name)
            disp, tier = matcher.match(model_name, extra=(model,))
            agg = defaultdict(list)
            for run, res in results:
                evals = ((res.get("stats") or {}).get("evals")) or {}
                for ev in evals.values():
                    rstats = ((ev.get("reward_stats") or {}).get("reward")) or {}
                    for val, trials in rstats.items():
                        try:
                            v = float(val)
                        except Exception:
                            continue
                        for tname in trials:
                            task = str(tname).split("__")[0]
                            agg[task].append(v)
            if disp is None:
                unmatched[f"tb2:{model_name}"] += len(agg)
                continue
            for task, vals in agg.items():
                rows.append((disp, model_name, "terminal_bench_2",
                             f"{agent}|trials{len(vals)}", task, "",
                             sum(vals) / len(vals), tier))


def run_terminal_bench(matcher):
    failures, rows, unmatched = [], [], Counter()
    models_seen = set()
    for fn in (run_tb1, run_tb2):
        try:
            fn(matcher, rows, unmatched, models_seen, failures)
        except Exception as e:
            failures.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    df = finalize(rows)
    rep = source_report(df, models_seen, unmatched, failures)
    rep["notes"] = ("TB1: label = mean is_resolved over trials, instruction from results.json; "
                    "TB2: label = mean reward over trials listed in run result.json reward_stats "
                    "(errored/unlisted trials excluded), item_text unavailable; "
                    "condition = agent|trialsN")
    write_source("terminal_bench", df, rep)


# ---------------------------------------------------------------- merge

WAVE2 = ["livebench", "alpacaeval", "taubench", "arcagi", "biggen", "terminal_bench"]


def merge():
    rpath = os.path.join(OUT, "report2.json")
    report = {"sources": {}, "overall": {}}
    if os.path.exists(rpath):
        report = json.load(open(rpath))
    total_rows, all_subjects, files = 0, set(), []
    for name in WAVE2:
        frag = os.path.join(OUT, f"r2_{name}.json")
        if os.path.exists(frag):
            report["sources"][name] = json.load(open(frag))
        else:
            report["sources"].setdefault(name, {"fatal_error": "no report fragment produced"})
        pq = os.path.join(OUT, f"{name}.parquet")
        if os.path.exists(pq):
            d = pd.read_parquet(pq, columns=["subject_match"])
            total_rows += len(d)
            all_subjects |= set(d["subject_match"].unique())
            files.append(f"{name}.parquet")
    report["overall"] = {
        "wave2_total_rows": total_rows,
        "wave2_matched_subjects": len(all_subjects),
        "wave2_parquet_files": files,
    }
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["overall"], indent=2))


RUNNERS = {
    "livebench": run_livebench,
    "alpacaeval": run_alpacaeval,
    "taubench": run_taubench,
    "arcagi": run_arcagi,
    "biggen": run_biggen,
    "terminal_bench": run_terminal_bench,
}


def main():
    what = sys.argv[1]
    if what == "merge":
        merge()
        return
    subjects = pd.read_parquet("/tmp/subjects.parquet")
    matcher = Matcher(subjects)
    t0 = time.time()
    try:
        RUNNERS[what](matcher)
    except Exception as e:
        traceback.print_exc()
        with open(os.path.join(OUT, f"r2_{what}.json"), "w") as f:
            json.dump({"fatal_error": f"{type(e).__name__}: {e}"}, f, indent=2)
        sys.exit(1)
    print(f"[{what}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
