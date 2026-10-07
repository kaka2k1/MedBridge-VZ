"""Learning-to-rank (LightGBM) thay cho RRF: học thứ tự doc từ nhãn thật (chunk trúng ⇒ doc gold) của các lần chấm.

    python scripts/ltr.py --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy \
        --dense data/cache/dense_gn_top300_v2.json --dense2 data/cache/kaggle_bgem3_out/dense_chunks_top.json \
        --dense3 data/cache/dense_chunks_top_v2.json --out data/cache/ltr_order.json
In AUC/P@k (CV 5 fold theo câu) của RRF và LTR trên doc có nhãn, rồi train trên tất cả và ghi thứ tự doc mới cho 1200 câu
→ dùng với `fuse_dense.py ... --order data/cache/ltr_order.json`.
Nhãn: doc từng được nộp kèm chunk; 1 nếu có chunk trúng, 0 nếu không (nhiễu: doc gold nhưng chunk trượt ⇒ 0).
"""
import argparse
import collections
import glob
import json
import sys
from urllib.parse import urlparse

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from rerank_vi import load_preds, pairs  # noqa: E402

MISS = 1000


def add_prf(X, emb, col, pre):
    """cos(doc, trọng tâm top-k doc theo cột col) – cùng chủ đề với nhóm đầu ⇒ dễ là gold."""
    for k in (5, 20):
        X[f"{pre}{k}"] = 0.0
    for q, g in X.groupby("qid"):
        top = g.sort_values(col, ascending=False).doc.tolist()
        V = np.stack([emb[d] if d in emb else np.zeros(1024, np.float32) for d in g.doc])
        for k in (5, 20):
            c = [emb[d] for d in top[:k] if d in emb]
            if c:
                c = np.mean(c, 0); c /= np.linalg.norm(c) + 1e-9
                X.loc[g.index, f"{pre}{k}"] = V @ c


def labels():
    pos, seen = collections.defaultdict(set), collections.defaultdict(set)
    for f in glob.glob("data/eval/details_*.jsonl"):
        for l in open(f):
            j = json.loads(l); q = int(j["id"])
            for p in j["chunk"]["predictions"]:
                d = p["doc_id"].split(":")[1]; seen[q].add(d)
                if p["relevant"]:
                    pos[q].add(d)
    return pos, seen


def build(a):
    P = pairs(load_preds(a.preds), a.top)
    df = pd.DataFrame(P, columns=["qid", "q", "r", "doc", "text"]).assign(s=np.load(a.scores))
    srcs = {k: json.load(open(f)) for k, f in (("d1", a.dense), ("d2", a.dense2), ("d3", a.dense3)) if f}
    url = pd.read_parquet("data/ViBioMIR/links_corpus.parquet").set_index("id").url
    rows = []
    for qid, g in df.groupby("qid"):
        n = len(g)
        f = 0.85 * g.s.rank(ascending=False) / n + 0.15 * (g.r + 1) / n
        g = g.assign(f=f).sort_values("f")
        feat = collections.defaultdict(dict)
        for i, (d, s, r) in enumerate(zip(g.doc, g.s, g.r)):
            feat[d].update(bm_rank=r, gn_rank=i, gn_s=s)
        qf = {}
        for k, src in srcs.items():
            v = src.get(str(qid), [])
            qf[f"{k}_s50"] = v[49][0] if len(v) > 49 else 0
            qf[f"{k}_s0"] = v[0][0] if v else 0
            for i, x in enumerate(v):
                feat[str(x[1])].update({f"{k}_rank": i, f"{k}_s": x[0], f"{k}_gap": v[0][0] - x[0]})
        for d, ft in feat.items():
            rr = sum(w / (60 + 1 + ft.get(c, 1e9)) for c, w in (("gn_rank", 1), ("d1_rank", 2), ("d2_rank", 1), ("d3_rank", 2)))
            dom = urlparse(url.get(int(d), "")).netloc.replace("www.", "")
            rows.append(dict(qid=int(qid), doc=d, rrf=rr, dom=dom, qlen=len(g.q.iloc[0]),
                             nsrc=sum(c in ft for c in ("gn_rank", "d1_rank", "d2_rank", "d3_rank")), **qf, **ft))
    X = pd.DataFrame(rows)
    # đặc trưng từ vựng: trùng âm tiết câu hỏi–tiêu đề, độ dài doc
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    need = set(X.doc); meta = {}
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "title", "text"])
        ids = [str(x) for x in t.column("doc_id").to_pylist()]
        ln = pc.utf8_length(t.column("text")).to_pylist(); ti = t.column("title").to_pylist()
        for d, a_, b_ in zip(ids, ti, ln):
            if d in need and d not in meta:
                meta[d] = (a_ or "", b_ or 0)
        del t
    # PRF: cos(doc, trọng tâm top-k doc theo RRF) bằng embedding GreenNode doc-level đã có (data/cache/dense_gn)
    emb = {}
    for f in sorted(glob.glob("data/cache/dense_gn/part*.npy")):
        ids = json.load(open(f.replace(".npy", ".ids.json")))
        idx = [i for i, d in enumerate(ids) if d in need]
        if idx:
            E = np.load(f, mmap_mode="r")[idx].astype(np.float32)
            emb.update({ids[i]: E[j] for j, i in enumerate(idx)})
    add_prf(X, emb, "rrf", "prf")
    if a.e5:   # model thứ 2 (multilingual-e5-large-instruct, mã hoá trên Colab: colab/encode_e5.py) – bổ sung GreenNode (AUC ghép .799)
        ids = json.load(open(f"{a.e5}/e5_ids.json")); E = np.load(f"{a.e5}/e5_docs.npy")
        e5 = {d: E[i].astype(np.float32) for i, d in enumerate(ids)}
        QE = dict(zip(json.load(open(f"{a.e5}/e5_qids.json")), np.load(f"{a.e5}/e5_queries.npy").astype(np.float32)))
        X["e5_s"] = [float(e5[d] @ QE[q]) if d in e5 else 0.0 for q, d in zip(X.qid, X.doc)]
        X["e5_rank"] = X.groupby("qid").e5_s.rank(ascending=False)
        X["e5_gap"] = X.groupby("qid").e5_s.transform("max") - X.e5_s
        add_prf(X, e5, "e5_s", "e5prf")
    tok = lambda s: set(s.lower().replace(",", " ").replace(".", " ").replace("?", " ").split())
    qtok = {q: tok(t) for q, t in df.groupby("qid").q.first().items()}
    X["doc_len"] = [meta.get(d, ("", 0))[1] for d in X.doc]
    tt = [tok(meta.get(d, ("", 0))[0]) for d in X.doc]
    X["title_ov"] = [len(t & qtok[q]) / (len(t) + 1) for t, q in zip(tt, X.qid)]
    X["title_cov"] = [len(t & qtok[q]) / (len(qtok[q]) + 1) for t, q in zip(tt, X.qid)]
    for c in X.columns:
        if c.endswith("_rank"):
            X[c] = X[c].fillna(MISS)
    return X, emb


def metrics(X, col):
    aucs, p10, p50 = [], [], []
    for _, g in X.groupby("qid"):
        y, s = g.y.values, g[col].values
        if 0 < y.sum() < len(y):
            pos, neg = s[y == 1], s[y == 0]
            aucs.append((pos[:, None] > neg[None, :]).mean())
        o = np.argsort(-s); p10.append(y[o[:10]].mean()); p50.append(y[o[:50]].mean())
    return dict(auc=round(float(np.mean(aucs)), 4), p10=round(float(np.mean(p10)), 4), p50=round(float(np.mean(p50)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--scores", required=True); ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--dense", required=True); ap.add_argument("--dense2", default=""); ap.add_argument("--dense3", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--e5", default="", help="thư mục e5_docs.npy/e5_ids.json/e5_queries.npy/e5_qids.json")
    ap.add_argument("--stages", type=int, default=1, help=">1: thêm PRF theo điểm LTR vòng trước")
    a = ap.parse_args()
    X, emb = build(a)
    pos, seen = labels()
    # tiên nghiệm tên miền: tính trong từng fold (chỉ từ câu train) để tránh rò rỉ
    X["lab"] = [d in seen.get(q, ()) for q, d in zip(X.qid, X.doc)]
    X["y"] = [int(d in pos.get(q, ())) for q, d in zip(X.qid, X.doc)]
    feats = [c for c in X.columns if c not in ("qid", "doc", "dom", "lab", "y")] + ["dom_rate"]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1)

    def dom_rate(train):
        t = train.groupby("dom").y.agg(["sum", "count"])
        return ((t["sum"] + 0.08 * 50) / (t["count"] + 50)).to_dict()

    qs = np.array(sorted(X[X.lab].qid.unique())); rng = np.random.default_rng(0); rng.shuffle(qs)
    folds = np.array_split(qs, 5)

    def fit(X, feats):
        cv = []
        for k in range(5):
            te = X.qid.isin(folds[k]) & X.lab; tr = ~X.qid.isin(folds[k]) & X.lab
            dr = dom_rate(X[tr]); Xk = X.assign(dom_rate=X.dom.map(dr).fillna(0.08))
            m = lgb.train(params, lgb.Dataset(Xk.loc[tr, feats], Xk.loc[tr, "y"]), a.rounds)
            allq = X.qid.isin(folds[k])   # dự đoán out-of-fold cho MỌI ứng viên của câu trong fold (điểm public trung thực)
            cv.append(Xk[allq].assign(ltr=m.predict(Xk.loc[allq, feats])))
        cv = pd.concat(cv)
        print("RRF", metrics(cv[cv.lab], "rrf")); print("LTR", metrics(cv[cv.lab], "ltr"), flush=True)
        dr = dom_rate(X[X.lab]); X["dom_rate"] = X.dom.map(dr).fillna(0.08)
        m = lgb.train(params, lgb.Dataset(X.loc[X.lab, feats], X.loc[X.lab, "y"]), a.rounds)
        X["ltr"] = m.predict(X[feats])
        oof = cv.set_index(["qid", "doc"]).ltr
        X["ltr"] = [oof.get((q, d), v) for q, d, v in zip(X.qid, X.doc, X.ltr)]   # câu có nhãn: dùng điểm OOF
        return m

    m = fit(X, feats)
    for st in range(a.stages - 1):   # PRF lặp: trọng tâm theo điểm LTR của vòng trước (OOF ⇒ không rò nhãn)
        add_prf(X, emb, "ltr", f"prfL{st}_")
        feats = feats + [f"prfL{st}_5", f"prfL{st}_20"]
        m = fit(X, feats)
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)[:12]
    print("importance", [(f, int(v)) for v, f in imp])
    out = {int(q): g.sort_values("ltr", ascending=False).doc.tolist() for q, g in X.groupby("qid")}
    psum = {int(q): [float(g.ltr.sum()), float(g.ltr.nlargest(50).sum()), float(g.ltr.nlargest(200).sum())] for q, g in X.groupby("qid")}
    json.dump(psum, open(a.out.replace(".json", "_psum.json"), "w"))
    json.dump({int(q): g.ltr.sort_values(ascending=False).head(400).round(5).tolist() for q, g in X.groupby("qid")},
              open(a.out.replace(".json", "_probs.json"), "w"))   # xác suất đã sắp xếp (khớp thứ tự trong --out) cho cắt ngưỡng F2
    json.dump(out, open(a.out, "w")); print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
