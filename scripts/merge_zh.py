"""Trộn doc ZH (xác suất từ scripts/zh_ltr.py) vào bài VI đã có (zip, vd v65): mỗi câu gộp (doc VI, p LTR) ∪ (doc ZH, p ZH), sắp theo p,
số doc = argmax F2 kỳ vọng (như fuse_dense --ef2), chunk theo cùng thứ tự (VI: chunk sẵn trong zip gốc; ZH: nguyên văn text[:--zh-chars]), tối đa --chunk-max.
    python scripts/merge_zh.py --base submissions/v65.zip --order data/cache/ltrx7_out/ltr_order_oof34 --zh data/cache/zh_probs.parquet --out data/cache/tmp/v70_deflate.zip [--sim]
--sim: chỉ in F2 docs mô phỏng (nhãn hiện có, doc chưa thấy = sai) cho bài gốc vs bài trộn.
"""
import argparse, json, sys, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--order", required=True); ap.add_argument("--zh", required=True); ap.add_argument("--out", default="")
ap.add_argument("--ef2", type=float, default=1.9); ap.add_argument("--lo", type=int, default=30); ap.add_argument("--hi", type=int, default=300)
ap.add_argument("--chunk-max", type=int, default=240); ap.add_argument("--zh-chars", type=int, default=3000); ap.add_argument("--zh-top", type=int, default=60)
ap.add_argument("--zh-scale", type=float, default=1.0, help="nhân xác suất ZH (hiệu chỉnh)"); ap.add_argument("--sim", action="store_true")
a = ap.parse_args()
z = zipfile.ZipFile(a.base); B = {r["id"]: r for r in json.loads(z.read(z.namelist()[0]))}
od = json.load(open(a.order + ".json")); pr = json.load(open(a.order + "_probs.json"))
Z = pd.read_parquet(a.zh); Z["doc_id"] = Z.doc_id.astype(str)
ZG = {q: g.sort_values("p", ascending=False).head(a.zh_top) for q, g in Z.groupby("qid")}
need = set(Z.doc_id)
TX = {}
if not a.sim:
    D = pd.read_parquet("data/cache/kg_zhdocs/zh_docs.parquet", columns=["doc_id", "text"]); D["doc_id"] = D.doc_id.astype(str)
    TX = dict(zip(D.doc_id, D.text.fillna("")))
pos, seen = labels()
gold = {int(json.loads(l)["id"]): json.loads(l)["document"]["gold"] for l in open("data/eval/details_6294_v65.jsonl")}


def f2(docs, q):
    g = gold.get(q, 0)
    if not g or not docs: return 0.0
    r = sum(d in pos[q] for d in docs); P, R = r / len(docs), r / g
    return 5 * P * R / (4 * P + R) if r else 0.0


out, s0, s1, nzh = [], [], [], []
for q in sorted(B):
    b = B[q]; vi = [(d, p, "vi") for d, p in zip(od.get(str(q), []), pr.get(str(q), []))]
    zg = ZG.get(q)
    zh = [(d, min(1.0, p * a.zh_scale), "zh") for d, p in zip(zg.doc_id, zg.p)] if zg is not None else []
    allp = sorted(vi + zh, key=lambda x: -x[1])[:400]
    p = np.array([x[1] for x in allp]); n = np.arange(1, len(p) + 1)
    nd = int(np.clip(np.argmax(5 * np.cumsum(p) / (4 * a.ef2 * p.sum() + n)) + 1, a.lo, a.hi)) if len(p) else 0
    sel = allp[:nd]; nzh.append(sum(x[2] == "zh" for x in sel))
    if q in seen:
        s0.append(f2([str(d) for d in b["relevant_docs"]], q)); s1.append(f2([x[0] for x in sel], q))
    if a.sim: continue
    vch = {str(c["doc_id"]): c["chunk_text"] for c in b["relevant_chunks"]}
    ch = []
    for d, _, lang in sel:
        t = vch.get(d) if lang == "vi" else TX.get(d, "")[:a.zh_chars]
        if t: ch.append({"doc_id": int(d), "chunk_text": t})
        if len(ch) >= a.chunk_max: break
    out.append({"id": q, "relevant_docs": [int(x[0]) for x in sel], "relevant_chunks": ch})
print(f"F2 docs mô phỏng (public, nhãn hiện có): gốc {np.mean(s0):.4f} → trộn ZH {np.mean(s1):.4f}  | doc ZH/câu chọn: TB {np.mean(nzh):.1f}")
if not a.sim and a.out:
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zz:
        zz.writestr("submission.json", json.dumps(out, ensure_ascii=False))
    print("wrote", a.out, len(out), "| chunk ZH:", sum(1 for r in out for c in r["relevant_chunks"] if str(c["doc_id"]) in need))
