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
    tok = lambda s: set(s.lower().replace(",", " ").replace(".", " ").replace("?", " ").split())
    qtok = {q: tok(t) for q, t in df.groupby("qid").q.first().items()}
    X["doc_len"] = [meta.get(d, ("", 0))[1] for d in X.doc]
    tt = [tok(meta.get(d, ("", 0))[0]) for d in X.doc]
    X["title_ov"] = [len(t & qtok[q]) / (len(t) + 1) for t, q in zip(tt, X.qid)]
    X["title_cov"] = [len(t & qtok[q]) / (len(qtok[q]) + 1) for t, q in zip(tt, X.qid)]
    for c in X.columns:
        if c.endswith("_rank"):
            X[c] = X[c].fillna(MISS)
    return X


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
    a = ap.parse_args()
    X = build(a)
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
    folds = np.array_split(qs, 5); cv = []
    for k in range(5):
        te = X.qid.isin(folds[k]) & X.lab; tr = ~X.qid.isin(folds[k]) & X.lab
        dr = dom_rate(X[tr]); Xk = X.assign(dom_rate=X.dom.map(dr).fillna(0.08))
        m = lgb.train(params, lgb.Dataset(Xk.loc[tr, feats], Xk.loc[tr, "y"]), a.rounds)
        allq = X.qid.isin(folds[k])   # dự đoán out-of-fold cho MỌI ứng viên của câu trong fold (điểm public trung thực)
        cv.append(Xk[allq].assign(ltr=m.predict(Xk.loc[allq, feats])))
    cv = pd.concat(cv)
    print("RRF", metrics(cv[cv.lab], "rrf")); print("LTR", metrics(cv[cv.lab], "ltr"))
    dr = dom_rate(X[X.lab]); X["dom_rate"] = X.dom.map(dr).fillna(0.08)
    m = lgb.train(params, lgb.Dataset(X.loc[X.lab, feats], X.loc[X.lab, "y"]), a.rounds)
    X["ltr"] = m.predict(X[feats])
    oof = cv.set_index(["qid", "doc"]).ltr
    X["ltr"] = [oof.get((q, d), v) for q, d, v in zip(X.qid, X.doc, X.ltr)]   # câu có nhãn: dùng điểm OOF
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)[:12]
    print("importance", [(f, int(v)) for v, f in imp])
    out = {int(q): g.sort_values("ltr", ascending=False).doc.tolist() for q, g in X.groupby("qid")}
    psum = {int(q): [float(g.ltr.sum()), float(g.ltr.nlargest(50).sum()), float(g.ltr.nlargest(200).sum())] for q, g in X.groupby("qid")}
    json.dump(psum, open(a.out.replace(".json", "_psum.json"), "w"))
    json.dump(out, open(a.out, "w")); print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
