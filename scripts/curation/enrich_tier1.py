#!/usr/bin/env python
"""Post-hoc item_text enrichment for eval_arena + swebench parquets via HF datasets-server."""
import json
import sys
import time

import pandas as pd
import requests

API = "https://datasets-server.huggingface.co/rows"
MAX_TEXT = 4000


def fetch_all(dataset, config="default", split="test", limit=None):
    rows, offset = [], 0
    while True:
        r = requests.get(API, params={"dataset": dataset, "config": config,
                                      "split": split, "offset": offset, "length": 100},
                         timeout=60)
        r.raise_for_status()
        batch = r.json()["rows"]
        if not batch:
            break
        rows.extend(b["row"] for b in batch)
        offset += len(batch)
        if limit and offset >= limit:
            break
        if len(batch) < 100:
            break
        time.sleep(0.2)
    return rows


def main():
    texts = {}  # (benchmark-ish key, item_id) -> text  ; keyed per benchmark name

    maps = {}

    # evalplus
    try:
        he = fetch_all("evalplus/humanevalplus")
        maps["evalplus_humaneval"] = {r["task_id"]: r["prompt"][:MAX_TEXT] for r in he}
        print("humanevalplus", len(he))
    except Exception as e:
        print("humanevalplus FAILED", e)
    try:
        mb = fetch_all("evalplus/mbppplus")
        maps["evalplus_mbpp"] = {r["task_id"]: r["prompt"][:MAX_TEXT] for r in mb}
        print("mbppplus", len(mb))
    except Exception as e:
        print("mbppplus FAILED", e)

    # cruxeval: id 'sample_N' -> eval-arena 'CRUXEval-input/N' & 'CRUXEval-output/N'
    try:
        cx = fetch_all("cruxeval-org/cruxeval")
        cmap = {}
        for r in cx:
            n = r["id"].split("_")[-1]
            code, inp, outp = r["code"], r["input"], r["output"]
            cmap[f"CRUXEval-input/{n}"] = f"{code}\n# given output: {outp}"[:MAX_TEXT]
            cmap[f"CRUXEval-output/{n}"] = f"{code}\n# given input: {inp}"[:MAX_TEXT]
        maps["cruxeval"] = cmap
        print("cruxeval", len(cx))
    except Exception as e:
        print("cruxeval FAILED", e)

    # SWE-bench Verified problem statements
    try:
        sb = fetch_all("princeton-nlp/SWE-bench_Verified")
        smap = {r["instance_id"]: r["problem_statement"][:MAX_TEXT] for r in sb}
        for b in ("swebench_verified_evalarena", "swebench_verified", "swebench_bashonly"):
            maps[b] = smap
        print("swebench_verified", len(sb))
    except Exception as e:
        print("swebench_verified FAILED", e)

    for pq in ("/tmp/curation/eval_arena.parquet", "/tmp/curation/swebench_experiments.parquet"):
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(pq, "read FAILED", e)
            continue
        filled = 0
        for bench, mp in maps.items():
            sel = df["benchmark"] == bench
            if not sel.any():
                continue
            new = df.loc[sel, "item_id"].map(mp).fillna("")
            filled += int((new != "").sum())
            df.loc[sel, "item_text"] = new
        df.to_parquet(pq, index=False)
        n = len(df)
        print(f"{pq}: filled item_text for {filled}/{n} rows")


if __name__ == "__main__":
    main()
