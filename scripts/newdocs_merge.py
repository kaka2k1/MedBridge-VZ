"""Gộp output kernel kaggle_newdocs/ (đặc trưng cho doc mới crawl) vào các file cache của pipeline → bản *_v3.

    python scripts/newdocs_merge.py --kout data/cache/newdocs_out --docs <dir có new_docs.parquet> --tag 91000
Sau đó chạy ltr.py / fuse_dense.py với các đường dẫn *_v3 (in ra cuối).
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

OLD = {"d1": "data/cache/dense_gn_top300_v2.json", "d2": "data/cache/kaggle_bgem3_out/dense_chunks_top.json",
       "d3": "data/cache/dense_chunks_top_v2.json"}
NEW = {"d1": "data/cache/dense_gn_top300_v3.json", "d2": "data/cache/dense_bgem3_chunks_v3.json", "d3": "data/cache/dense_chunks_top_v3.json"}
RR = {"rr": "rr_bge_all", "rr2": "rr_win", "rr3": "ce_ft", "rr4": "ce_ftwin", "gnft": "gnft"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kout", required=True); ap.add_argument("--docs", required=True); ap.add_argument("--tag", default="91000")
    a = ap.parse_args()
    K = a.kout
    # 1) doc mới vào data/docs (fuse_dense / ltr đọc văn bản & tiêu đề từ đây)
    dst = f"data/docs/corpus-{a.tag}.parquet"
    shutil.copy(f"{a.docs}/new_docs.parquet", dst); print("docs ->", dst)
    # 2) embedding GN doc-level cho PRF (ltr.py đọc mọi data/cache/dense_gn/part*.npy)
    np.save(f"data/cache/dense_gn/part{a.tag}.npy", np.load(f"{K}/gn_docs.npy"))
    shutil.copy(f"{K}/gn_ids.json", f"data/cache/dense_gn/part{a.tag}.ids.json")
    # 3) gộp list dense: top-300 theo điểm trên (cũ ∪ mới)
    for k in ("d1", "d2", "d3"):
        old, new = json.load(open(OLD[k])), json.load(open(f"{K}/{k}_new.json"))
        out = {}
        for q in old:
            best = {}
            for x in old[q] + new.get(q, []):
                if str(x[1]) not in best or x[0] > best[str(x[1])][0]:
                    best[str(x[1])] = x
            out[q] = sorted(best.values(), key=lambda x: -x[0])[:300]
        nn = sum(str(x[1]) in {str(y[1]) for y in new.get(q, [])} for q in out for x in out[q])
        json.dump(out, open(NEW[k], "w"), ensure_ascii=False); print(k, "->", NEW[k], f"new-doc entries {nn} ({nn / len(out):.1f}/câu)")
    # 4) e5: nối embedding doc mới vào bản cũ
    os.makedirs("data/cache/e5_v3", exist_ok=True)
    ids = json.load(open("data/cache/e5/e5_ids.json")) + json.load(open(f"{K}/e5_ids.json"))
    np.save("data/cache/e5_v3/e5_docs.npy", np.concatenate([np.load("data/cache/e5/e5_docs.npy"), np.load(f"{K}/e5_docs.npy")]))
    json.dump(ids, open("data/cache/e5_v3/e5_ids.json", "w"))
    for f in ("e5_queries.npy", "e5_qids.json"):
        shutil.copy(f"data/cache/e5/{f}", f"data/cache/e5_v3/{f}")
    # 5) điểm cross-encoder / GN fine-tune cho cặp (câu, doc mới) – model train trên tất cả nhãn (kaggle_savemodels)
    P = pd.read_parquet(f"{K}/pairs.parquet"); P["doc_id"] = P.doc_id.astype(str)
    for col, name in RR.items():
        R = pd.read_parquet(f"data/cache/{name}.parquet")[["qid", "doc_id", "s"]]; R["doc_id"] = R.doc_id.astype(str)
        pd.concat([R, P[["qid", "doc_id"]].assign(s=P[col])]).drop_duplicates(["qid", "doc_id"]).to_parquet(f"data/cache/{name}_v3.parquet", index=False)
    print("pairs", len(P))
    v3 = lambda n: f"data/cache/{n}_v3.parquet"
    print(f"ltr.py --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy --dense {NEW['d1']} --dense2 {NEW['d2']} "
          f"--dense3 {NEW['d3']} --e5 data/cache/e5_v3 --rr {v3('rr_bge_all')} --rr2 {v3('rr_win')} --rr3 {v3('ce_ft')} --rr4 {v3('ce_ftwin')} "
          f"--gnft {v3('gnft')} --stages 2 --seeds 5 --out data/cache/ltr_order_oof17.json")


if __name__ == "__main__":
    main()
