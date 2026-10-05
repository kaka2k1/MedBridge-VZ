"""Hợp nhất ứng viên: (BM25 → GreenNode chunk rerank) ⊕ (GreenNode dense toàn corpus) bằng RRF, rồi ghi zip nộp.

    python scripts/fuse_dense.py --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy --top 200 \
        --dense data/cache/dense_gn_top300.json --w-dense 1.0 --n-docs 120 --n-chunks 80 --out submissions/v9.zip
    (--eval-only: chỉ đếm doc đúng-đã-biết trong top-n từ data/eval/details_*.jsonl, không ghi zip)
"""
import argparse
import glob
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "scripts")
from chunk_v2 import select_chunks  # noqa: E402
from rerank_vi import load_preds, pairs  # noqa: E402


def fused_lists(a):
    P = pairs(load_preds(a.preds), a.top)
    S = np.load(a.scores)
    df = pd.DataFrame(P, columns=["qid", "q", "r", "doc", "text"]).assign(s=S)
    dense = json.load(open(a.dense))
    dense2 = json.load(open(a.dense2)) if a.dense2 else {}
    plans = {int(json.loads(l)["id"]): json.loads(l) for l in open("data/cache/query_plans.jsonl")}
    out = {}
    for qid, g in df.groupby("qid"):
        n = len(g)
        f = a.alpha * g.s.rank(ascending=False) / n + (1 - a.alpha) * (g.r + 1) / n
        g = g.assign(f=f).sort_values("f")
        rr = {}
        for i, d in enumerate(g.doc):
            rr[d] = rr.get(d, 0) + 1 / (a.k + i + 1)
        for i, (_, d) in enumerate(dense.get(str(qid), [])[:a.dense_k]):
            rr[str(d)] = rr.get(str(d), 0) + a.w_dense / (a.k + i + 1)
        for i, (_, d) in enumerate(dense2.get(str(qid), [])[:a.dense_k]):
            rr[str(d)] = rr.get(str(d), 0) + a.w_dense2 / (a.k + i + 1)
        docs = sorted(rr, key=lambda d: -rr[d])
        out[int(qid)] = (docs, dict(zip(g.doc, g.text)), g.q.iloc[0], plans.get(int(qid)))
    return out


def known_pos():
    pos = {}
    for f in glob.glob("data/eval/details_*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            for p in d["chunk"]["predictions"]:
                if p["relevant"]:
                    pos.setdefault(int(d["id"]), set()).add(p["doc_id"].split(":")[1])
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--scores", required=True)
    ap.add_argument("--top", type=int, default=200); ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--dense", required=True); ap.add_argument("--dense-k", type=int, default=300)
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--dense2", default=""); ap.add_argument("--w-dense2", type=float, default=1.0); ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--n-docs", type=int, default=120); ap.add_argument("--n-chunks", type=int, default=80)
    ap.add_argument("--out"); ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adaptive", type=float, default=0, help=">0: số doc/câu = hệ số × số gold dự đoán (40..200)")
    ap.add_argument("--chunk-ratio", type=float, default=0.67)
    a = ap.parse_args()
    L = fused_lists(a)
    if a.eval_only:
        pos = known_pos()
        for n in (60, 120):
            print(f"w_dense={a.w_dense} known-pos in top{n}:",
                  round(sum(len(set(L[q][0][:n]) & pos.get(q, set())) for q in L) / len(pos), 3))
        return
    need = {d for q, (docs, ch, *_ ) in L.items() for d in docs[:130] if d not in ch}
    texts = {}
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "text"]).to_pandas()
        t = t[t.doc_id.astype(str).isin(need)]
        texts.update(zip(t.doc_id.astype(str), t.text))
    nd, nc = {}, {}
    if a.adaptive:
        # số doc theo từng câu: điểm dense hạng 50 → phân vị → số gold dự đoán (map từ public details) × hệ số
        mp = json.load(open("data/cache/gold_count_map.json")); xs, ys = np.array(mp["xs"]), np.array(mp["ys"])
        dn = json.load(open(a.dense))
        for qid in L:
            v = dn.get(str(qid), [])
            s50 = v[49][0] if len(v) > 49 else (v[-1][0] if v else 0)
            pred = np.quantile(ys, min(1.0, np.searchsorted(xs, s50) / len(xs)))
            nd[qid] = int(np.clip(round(a.adaptive * pred), 40, 200))
            nc[qid] = int(np.clip(round(nd[qid] * a.chunk_ratio), 30, 130))
    res = []
    for qid, (docs, ch, qtext, plan) in sorted(L.items()):
        n_docs, n_chunks = nd.get(qid, a.n_docs), nc.get(qid, a.n_chunks)
        chunks = []
        for d in docs[:n_chunks]:
            c = ch.get(d)
            if not c and d in texts:
                sel = select_chunks(qtext, plan, texts[d], "vi", {"target_chars": 1600})
                c = sel[0] if sel else ""
            if c:
                chunks.append({"doc_id": int(d), "chunk_text": c})
        res.append({"id": qid, "relevant_docs": [int(d) for d in docs[:n_docs]], "relevant_chunks": chunks})
    ids = set(pd.read_parquet("data/ViBioMIR/query.parquet").id.astype(int))
    res += [{"id": i, "relevant_docs": [], "relevant_chunks": []} for i in sorted(ids - {r["id"] for r in res})]
    res.sort(key=lambda e: e["id"])
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("submission.json", json.dumps(res, ensure_ascii=False))
    print("wrote", a.out, len(res))


if __name__ == "__main__":
    main()
