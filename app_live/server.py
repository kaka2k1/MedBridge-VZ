"""ViBioMIR Live – máy chủ tra cứu bằng chứng y khoa chạy trực tiếp trên câu hỏi bất kỳ.

Pipeline (bản trực tuyến rút gọn của hệ thống thi):
  1. Tìm ứng viên VI: GreenNode dense (738k trang, tìm chính xác) ∪ FTS5 theo tiêu đề (từ khoá)
  2. Dịch câu hỏi VI→ZH (Qwen3-4B-Instruct, 4-bit) → bge-m3 dense trên 222k trang ZH (câu gốc + câu dịch)
  3. Chấm lại bằng cross-encoder bge-reranker-v2-m3 trên cửa sổ đoạn khớp nhất của mỗi tài liệu
  4. Trộn VI + ZH theo xác suất (ZH × 0.9), chọn số tài liệu n tối đa F2 kỳ vọng (c = 1.9)
Chạy:  .venv/bin/python app_live/server.py   → mở http://localhost:8765
"""
import asyncio, glob, json, math, os, queue, re, sqlite3, time, threading
import numpy as np, torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from urllib.parse import urlparse

R = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(R)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRANSLATE = os.environ.get("VIBIOMIR_TRANSLATE", "1") == "1"
K_VI, K_FTS, K_ZH, CE_VI, CE_ZH, C_EF2, ZH_SCALE, CE_SHIFT, N_MIN = 200, 40, 120, 60, 40, 1.9, 0.9, 3.0, 10
STOP = set("tôi em cho hỏi bị có không là và của thì này được bác sĩ ạ vậy làm gì nào như với các những một bệnh khi nên cần sao thế ra vào lại đã đang sẽ rồi mà nhưng hay hoặc thể theo tiếp bước bao nhiêu thường xuyên hiện nay do ơi mình bạn nhé".split())
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------- tải chỉ mục & mô hình ----------
log("loading VI embeddings (GreenNode)…")
ids_vi, parts = [], []
for f in sorted(glob.glob(f"{ROOT}/data/cache/dense_gn/*.npy")):
    ids_vi += [int(x) for x in json.load(open(f[:-4] + ".ids.json"))]; parts.append(np.load(f))
E_VI = torch.from_numpy(np.concatenate(parts)); del parts            # CPU fp16, nhân theo khối
ids_vi = np.array(ids_vi); log("VI", E_VI.shape)
log("loading ZH embeddings (bge-m3)…")
E_ZH = torch.from_numpy(np.load(f"{ROOT}/data/cache/zhfeat3_out/zh_emb.npy"))   # CPU fp16, nhân theo khối
ids_zh = np.array([int(x) for x in json.load(open(f"{ROOT}/data/cache/zhfeat3_out/zh_ids.json"))]); log("ZH", tuple(E_ZH.shape))
from sentence_transformers import SentenceTransformer, CrossEncoder
kw = {"model_kwargs": {"torch_dtype": torch.float16}} if DEV == "cuda" else {}
GN = SentenceTransformer("GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1", device=DEV, **kw); GN.max_seq_length = 256
M3 = SentenceTransformer(f"{ROOT}/data/models/bge-m3-st", device=DEV, **kw); M3.max_seq_length = 256
CE = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device=DEV, **kw)
QW = TOK = None
if TRANSLATE:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
    TOK = AutoTokenizer.from_pretrained(name); QW = AutoModelForCausalLM.from_pretrained(name, device_map=DEV)
    log("translator ready")
DBG = {}
DB = f"{R}/docs.sqlite"; LOCK = threading.Lock()
log("ready")

SYS_ZH = "你是专业的医学翻译。把用户给出的越南语医学问题准确翻译成简体中文，保留所有症状、疾病、药物、检查名称等医学术语的准确含义，不要回答问题，不要添加解释，只输出译文。"
def translate(q):
    if QW is None: return ""
    msgs = [{"role": "system", "content": SYS_ZH}, {"role": "user", "content": q}]
    x = TOK.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEV)
    with torch.no_grad(): y = QW.generate(x, max_new_tokens=160, do_sample=False, repetition_penalty=1.05)
    return TOK.decode(y[0, x.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]

def terms_vi(q):
    w = re.findall(r"\w+", q.lower()); t = {x for x in w if len(x) > 1 and x not in STOP}
    return t | {w[i] + " " + w[i + 1] for i in range(len(w) - 1) if not (w[i] in STOP and w[i + 1] in STOP)}
def terms_zh(z):
    s = re.sub(r"[^一-鿿]", "", z); return {s[i:i + 2] for i in range(len(s) - 1)}
def best_window(text, T, size):
    if len(text) <= size: return text, 0
    low = text.lower(); hit = np.zeros(len(text) + 1)
    for t in T:
        for m in re.finditer(re.escape(t), low): hit[m.start()] += len(t)
    pre = np.concatenate([[0], np.cumsum(hit)]); step = 25
    starts = np.arange(0, len(text) - size + 1, step); sc = pre[starts + size] - pre[starts]
    st = int(starts[int(np.argmax(sc))]) if len(starts) else 0
    k = text.rfind(" ", max(0, st - 40), st + 1); st = k + 1 if (k >= 0 and st) else st
    return text[st:st + size], st

def dense_top(E, ids, Q, k, B=50_000):
    """Tìm chính xác: E (N×d, CPU fp16) · Q (d×m) theo khối trên GPU, lấy max theo m câu truy vấn."""
    best_s, best_i = [], []; qh = Q.half() if DEV == "cuda" else Q.float()
    for a in range(0, E.shape[0], B):
        blk = E[a:a + B].to(DEV, non_blocking=True) if DEV == "cuda" else E[a:a + B].float()
        s = (blk @ qh).float(); s = s.max(dim=1).values if s.dim() == 2 else s
        v, i = torch.topk(s, min(k, len(s))); best_s.append(v.cpu()); best_i.append(i.cpu() + a); del blk, s
    s = torch.cat(best_s); i = torch.cat(best_i); o = torch.topk(s, k).indices
    return ids[i[o].numpy()], s[o].numpy()

def fts_vi(db, q, k):
    T = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 1 and t not in STOP]
    if not T: return []
    expr = " OR ".join('"' + t.replace('"', "") + '"' for t in T[:12])
    try:
        rid = [r[0] for r in db.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY rank LIMIT ?", (expr, 2 * k))]
        if not rid: return []
        got = dict(db.execute(f"SELECT rowid, id FROM d WHERE lang='vi' AND rowid IN ({','.join('?' * len(rid))})", rid).fetchall())
        return [got[r] for r in rid if r in got][:k]
    except sqlite3.OperationalError: return []

def search_iter(q):
    """Chạy pipeline, phát sự kiện tiến trình thật cho giao diện: stage(k, xong) → zh → done."""
    tm = {}; t0 = time.time()
    db = sqlite3.connect(DB)
    yield dict(ev="stage", k=0)
    with torch.no_grad():
        qe = GN.encode(q, normalize_embeddings=True, convert_to_tensor=True).float()
        v_ids, v_s = dense_top(E_VI, ids_vi, qe, K_VI); f_ids = fts_vi(db, q, K_FTS)
    rrf = {}
    for r, d in enumerate(v_ids.tolist()): rrf[d] = rrf.get(d, 0) + 2 / (60 + r)
    for r, d in enumerate(f_ids): rrf[d] = rrf.get(d, 0) + 1 / (60 + r)
    vi_c = sorted(rrf, key=rrf.get, reverse=True)[:CE_VI]; tm["Tìm tiếng Việt"] = time.time() - t0
    yield dict(ev="stage", k=1, t=round(tm["Tìm tiếng Việt"], 2), info=f"{len(rrf)} ứng viên (vector {len(v_ids)} + từ khóa {len(f_ids)})")
    t1 = time.time(); zh = translate(q) if TRANSLATE else ""; torch.cuda.empty_cache() if DEV == "cuda" else None; tm["Dịch sang tiếng Trung"] = time.time() - t1
    yield dict(ev="zh", zh=zh)
    yield dict(ev="stage", k=2, t=round(tm["Dịch sang tiếng Trung"], 2), info=zh or "bỏ qua")
    t2 = time.time()
    with torch.no_grad():
        Q = M3.encode([q] + ([zh] if zh else []), normalize_embeddings=True, convert_to_tensor=True).half()
        z_ids, _ = dense_top(E_ZH, ids_zh, Q.T, K_ZH)
    zh_c = z_ids[:CE_ZH].tolist(); tm["Tìm tiếng Trung"] = time.time() - t2
    yield dict(ev="stage", k=3, t=round(tm["Tìm tiếng Trung"], 2), info=f"{len(z_ids)} ứng viên, giữ {len(zh_c)}")
    t3 = time.time(); TV, TZ = terms_vi(q), terms_zh(zh)
    rows = {}
    for lang, cand in (("vi", vi_c), ("zh", zh_c)):
        for i in range(0, len(cand), 500):
            ch = cand[i:i + 500]
            for d, u, ti, tx in db.execute(f"SELECT id,url,title,text FROM d WHERE lang=? AND id IN ({','.join('?' * len(ch))})", (lang, *ch)): rows[(lang, d)] = (u or "", ti or "", tx or "")
    db.close()
    pairs, keys, snips = [], [], []
    for lang, cand in (("vi", vi_c), ("zh", zh_c)):
        for d in cand:
            if (lang, d) not in rows: continue
            u, ti, tx = rows[(lang, d)]; T = TV if lang == "vi" else TZ
            w, _ = best_window(tx, T, 900 if lang == "vi" else 500)
            pairs.append((q, f"{ti}\n{w}")); keys.append((lang, d))
            sn, st = best_window(tx, T, 460); snips.append((sn, st, len(tx)))
    with torch.no_grad(): logit = CE.predict(pairs, batch_size=12, show_progress_bar=False, activation_fn=torch.nn.Identity())
    p = 1 / (1 + np.exp(-(np.asarray(logit, dtype=np.float64) - CE_SHIFT)))   # hiệu chỉnh: logit của reranker lệch cao
    p = np.array([pi * (ZH_SCALE if k[0] == "zh" else 1) for pi, k in zip(p, keys)]); tm["Chấm lại (cross-encoder)"] = time.time() - t3
    o = np.argsort(-p); S = p.sum(); cum = np.cumsum(p[o]); n_all = len(o)
    ef = [5 * cum[n - 1] / (4 * C_EF2 * S + n) for n in range(1, n_all + 1)]
    n_star = int(np.argmax(ef[N_MIN - 1:]) + N_MIN) if n_all > N_MIN else n_all
    sel = o[:n_star]; nzh = sum(1 for j in sel if keys[j][0] == "zh")
    res = []
    for j in sel:
        lang, d = keys[j]; u, ti, tx = rows[(lang, d)]; sn, st, L = snips[j]
        # c = đoạn trích nguyên văn (không thêm dấu …); s = vị trí bắt đầu trong văn bản gốc
        res.append(dict(d=d, l=lang, p=round(float(p[j]), 3), c=sn.strip(), s=st, pre=st > 0, post=st + 460 < L,
                        u=u, t=ti[:160], h=urlparse(u).netloc.replace("www.", "") if u else ""))
    tm["Tổng"] = time.time() - t0
    LG = np.asarray(logit, dtype=np.float64); DBG.update(q=q, lq=np.percentile(LG, [0, 10, 25, 50, 75, 90, 100]).round(2).tolist())
    yield dict(ev="stage", k=4, t=round(tm["Chấm lại (cross-encoder)"], 2), info=f"{n_all} cặp, chọn {n_star}")
    yield dict(ev="done", q=q, zh=zh, n=n_star, nzh=nzh, cand=n_all, r=res, tm={k: round(v, 2) for k, v in tm.items()})

def search(q):
    for e in search_iter(q): pass
    e.pop("ev"); return e

# ---------- bộ đệm kết quả: câu hỏi lặp lại trả ngay, lưu ra đĩa để giữ qua lần khởi động lại ----------
from collections import OrderedDict
CACHE_F = f"{R}/cache.json"; CACHE_MAX = 300; CLOCK = threading.Lock()
try: _c = json.load(open(CACHE_F, encoding="utf-8"))
except Exception: _c = {}
RES_CACHE = OrderedDict(_c.get("res", {})); TR_CACHE = dict(_c.get("tr", {}))
def ckey(q): return re.sub(r"[^\w]+", " ", q.lower()).strip()
def cache_get(q):
    with CLOCK:
        v = RES_CACHE.get(ckey(q))
        if v is not None: RES_CACHE.move_to_end(ckey(q))
        return v
def cache_put(q, events):
    with CLOCK:
        RES_CACHE[ckey(q)] = events
        while len(RES_CACHE) > CACHE_MAX: RES_CACHE.popitem(last=False)
    cache_save()
def cache_save():
    with CLOCK: data = {"res": dict(RES_CACHE), "tr": dict(list(TR_CACHE.items())[-500:])}
    try:
        tmp = f"{CACHE_F}.{threading.get_ident()}.tmp"; json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False); os.replace(tmp, CACHE_F)
    except Exception as ex: log("cache save failed", repr(ex))

SYS_VI = "Bạn là biên dịch viên y khoa chuyên nghiệp. Dịch đoạn văn tiếng Trung do người dùng đưa ra sang tiếng Việt chính xác, tự nhiên, giữ đúng nghĩa các thuật ngữ y khoa (bệnh, triệu chứng, thuốc, xét nghiệm). Không giải thích, không thêm nội dung, chỉ xuất bản dịch."
LOOP = re.compile(r"(\S.{3,40}?)(?:\s*\1){3,}\s*$", re.S)
def translate_stream(text, stop=None):
    """Dịch ZH→VI, phát từng mảnh văn bản ngay khi sinh ra (TextIteratorStreamer)."""
    from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
    if text in TR_CACHE: yield TR_CACHE[text]; return
    stop = stop or threading.Event()
    class Stop(StoppingCriteria):
        def __call__(self, *a, **k): return stop.is_set()
    msgs = [{"role": "system", "content": SYS_VI}, {"role": "user", "content": text}]
    x = TOK.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEV)
    st = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=True, timeout=120)
    def run():
        with torch.no_grad():
            QW.generate(x, max_new_tokens=min(700, int(len(text) * 1.6) + 40), do_sample=False, repetition_penalty=1.05, streamer=st, stopping_criteria=StoppingCriteriaList([Stop()]))
    th = threading.Thread(target=run, daemon=True); th.start()
    full, sent, HOLD = "", 0, 180     # giữ lại đuôi chưa gửi để có thể cắt vòng lặp sinh ("Ví dụ: Ví dụ: …")
    try:
        for s in st:
            full += s; tail0 = max(0, len(full) - 260)
            m = LOOP.search(full, tail0)
            if m:
                full = full[:m.start()] + m.group(1) + " …"; stop.set(); break
            if len(full) - HOLD > sent: yield full[sent:len(full) - HOLD]; sent = len(full) - HOLD
        if len(full) > sent: yield full[sent:]
        with CLOCK: TR_CACHE[text] = full
        cache_save()
    finally:
        stop.set(); th.join()

app = FastAPI(title="ViBioMIR Live")
class Q(BaseModel): q: str
class TQ(BaseModel): text: str
def clean(q): return re.sub(r"\s+", " ", q).strip()[:600]

@app.post("/api/search")
def api(qq: Q):
    q = clean(qq.q)
    if len(q) < 4: return JSONResponse({"error": "Câu hỏi quá ngắn"}, 400)
    ev = cache_get(q)
    if ev is None:
        with LOCK: ev = list(search_iter(q))
        cache_put(q, ev)
    out = dict(ev[-1]); out.pop("ev"); return out

_END = object()
async def gpu_stream(make_iter, stop):
    """Chạy make_iter() trong luồng worker giữ LOCK (GPU), chuyển từng phần tử ra qua hàng đợi.
    Client ngắt kết nối → stop được đặt, worker dừng ở bước kế tiếp và tự nhả LOCK (không treo máy chủ)."""
    qu = queue.Queue()
    def work():
        try:
            with LOCK:
                if stop.is_set(): return
                for e in make_iter():
                    qu.put(e)
                    if stop.is_set(): break
        except Exception as ex:
            log("ERROR", repr(ex)); qu.put(ex)
        finally:
            qu.put(_END)
    threading.Thread(target=work, daemon=True).start()
    try:
        while True:
            e = await asyncio.to_thread(qu.get)
            if e is _END: break
            yield e
    finally:
        stop.set()

@app.post("/api/search/stream")
async def api_stream(qq: Q):
    """NDJSON: mỗi dòng một sự kiện tiến trình; dòng cuối ev=done chứa kết quả."""
    q = clean(qq.q)
    if len(q) < 4: return JSONResponse({"error": "Câu hỏi quá ngắn"}, 400)
    stop = threading.Event()
    hit = cache_get(q)
    async def gen():
        if hit is not None:                       # đã có: phát lại các bước thật thật nhanh, đánh dấu cached
            for e in hit:
                e = dict(e)
                if e["ev"] == "done": e["cached"] = True; log(f"cache n={e['n']} | {q[:60]}")
                else: await asyncio.sleep(0.08)
                yield json.dumps(e, ensure_ascii=False) + "\n"
            return
        if LOCK.locked(): yield json.dumps(dict(ev="queue"), ensure_ascii=False) + "\n"
        evs = []
        async for e in gpu_stream(lambda: search_iter(q), stop):
            if isinstance(e, Exception): e = dict(ev="error", error="Lỗi xử lý trên máy chủ: " + type(e).__name__)
            else:
                evs.append(e)
                if e["ev"] == "done": log(f"{e['tm']['Tổng']:.1f}s n={e['n']} zh={e['nzh']} | {q[:60]}"); cache_put(q, evs)
            yield json.dumps(e, ensure_ascii=False) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

@app.post("/api/translate")
async def api_translate(t: TQ):
    text = t.text.strip()[:1500]
    if QW is None: return JSONResponse({"error": "Chưa bật mô hình dịch (VIBIOMIR_TRANSLATE=0)"}, 503)
    if not text: return JSONResponse({"error": "Không có nội dung"}, 400)
    stop = threading.Event()
    async def gen():
        async for e in gpu_stream(lambda: translate_stream(text, stop), stop):
            if isinstance(e, Exception): yield " [lỗi dịch]"; break
            yield e
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-store"})

@app.get("/api/doc")
def api_doc(lang: str, id: int):
    db = sqlite3.connect(DB); r = db.execute("SELECT url,title,text FROM d WHERE lang=? AND id=?", (lang, id)).fetchone(); db.close()
    if not r: return JSONResponse({"error": "Không tìm thấy tài liệu"}, 404)
    u, ti, tx = r; tx = tx or ""
    return dict(u=u or "", t=ti or "", x=tx[:80000], cut=len(tx) > 80000, L=len(tx))

FAV = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="16" fill="#0E7C6B"/><path d="M27 14h10v13h13v10H37v13H27V37H14V27h13z" fill="#fff"/></svg>'
@app.get("/favicon.ico")
def fav(): return Response(FAV, media_type="image/svg+xml")
@app.get("/")
def index(): return FileResponse(f"{R}/index.html", headers={"Cache-Control": "no-cache"})
@app.get("/api/debug")
def dbg(): return DBG
@app.get("/api/health")
def health(): return {"ok": True, "vi_docs": int(E_VI.shape[0]), "zh_docs": int(E_ZH.shape[0]), "translate": TRANSLATE, "device": DEV,
                      "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU", "busy": LOCK.locked()}

if __name__ == "__main__":
    import uvicorn
    with LOCK: search("Chỉ số CA 125 cao nghi bệnh gì?"); log("warm-up done")
    port = int(os.environ.get("PORT", 8765))
    log(f"READY → mở trình duyệt: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
