# HANDOFF – R2AI2026 Truy hồi tài liệu y khoa đa ngôn ngữ (bàn giao cho agy / người tiếp theo)

> Cập nhật: 04/10/2026. Đọc file này TRƯỚC khi làm gì. Mục tiêu: vượt **0.3764** (baseline BM25 của BTC, đang đứng đầu).
> Đề bài đầy đủ: `R2AI2026_cuoc_thi.md` (lưu ý quy định mô hình mới nhất trên trang Overview: phát hành **trước 01/08/2026**, **≤ 15B** tham số, cấm LLM đóng).

---

## 0. TRẠNG THÁI MỚI NHẤT (04/10 ~19:45) – ĐỌC MỤC NÀY TRƯỚC

**Bài tốt nhất: v10 = 0.2097** (`submissions/v10_dense_w2.zip`, ID 4362, w_dense=2.0; w=1 → 0.2022, w=4 → 0.2068 ⇒ tối ưu quanh w=2). Cấu trúc như v9 (ID 4360 = RRF của [BM25 top200 → GreenNode chunk rerank] và [GreenNode DENSE top300 trên toàn bộ 625k doc VI], w_dense=1.0, 120 doc + 80 chunk; doc trúng 21.7/câu). Dense là bước nhảy lớn nhất → tiếp tục khai thác dense (tăng w, dense chunk-level, thêm model dense khác như bge-m3/Qwen3-Embedding để ensemble). BTC BM25 Baseline: 0.3764.

**Pipeline của bài tốt nhất (chỉ tiếng Việt, ZH đã chứng minh làm giảm điểm):**
```bash
# 1) BM25 VI top-300 ứng viên + chunk_v2 (~1700 ký tự) – CPU, ~5 phút (index có sẵn data/cache/bm25_vi)
nice .venv/bin/python scripts/pipeline_v2.py predict --out /tmp/x.zip --doc-max-vi 300 --doc-max-zh 0 \
    --chunk-chars-vi 1600 --dump-preds data/cache/vi_preds_300.jsonl
# 2) Chấm 200 ứng viên đầu bằng GreenNode-Embedding-Large-VN-Mixed-V1 (fp16, có checkpoint .partNNN.npy) – GPU ~80 phút trên RTX 3050
nice .venv/bin/python scripts/rerank_vi.py score --preds data/cache/vi_preds_300.jsonl --kind embed \
    --model GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1 --out data/cache/rr_gn_top200.npy --top 200 --batch 32
# 3) Trộn thứ hạng (alpha=0.85 cho model) → zip nộp
nice .venv/bin/python scripts/rerank_vi.py write --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy \
    --top 200 --alpha 0.85 --n-docs 120 --n-chunks 60 --out submissions/v5.zip
```

**Bảng nộp hôm 04/10:**
| ID | Bài | FINAL | Docs F2 | Chunks F2 | Ghi chú |
|---|---|---|---|---|---|
| 4281 | v3 VI20+ZH60 | 0.0553 | .073 | .038 | ZH BM25 (jieba, 780k trang) gần như 0 trúng |
| 4283 | v3 VI30+ZH40 | 0.0739 | | | |
| 4286 | v4 VI top100 BM25 | 0.1193 | .161 | .078 | gold chủ yếu là trang VI nằm sâu |
| 4324 | **v5 GreenNode rerank, 120 doc, 60 chunk** | **0.1507** | .194 | .108 | tốt nhất |
| 4326 | v6 = v5 nhưng 150 doc, 80 chunk | 0.1473 | .184 | .110 | 150 doc quá nhiều; 80 chunk tốt hơn 60 |
| 4339 | v8 = v5 với 120 doc + 80 chunk | 0.1520 | .194 | .110 | |
| 4360 | **v9 = v8 ⊕ dense GreenNode toàn corpus (RRF, w=1)** | **0.2022** | .258 | .147 | tốt nhất – offline (thiên BM25) đánh giá thấp dense |
| 4362 | **v10 = v9 với w_dense=2.0** | **0.2097** | .266 | .154 | **tốt nhất** |
| 4363 | v11 = w_dense=4.0 | 0.2068 | .261 | .152 | quá nhiều dense |
| 4335 | v7 BM25 "chuẩn" (âm tiết, không bigram, câu gốc, full text) top100, không rerank | 0.1009 | .137 | .065 | KÉM hơn v4 (0.1193) → giả thuyết BM25 chuẩn bị bác bỏ; giữ BM25 cũ |

**Phát hiện then chốt:**
- So sánh model offline bằng NHÃN THẬT (`scripts/eval_models.py`, nhãn từ details.jsonl): BM25 order AUC .55 / P@10 .12;
  bge-m3 .750/.207; bge-reranker-v2-m3 .765/.218; **GreenNode-VN .787/.232** (tốt nhất). Kết quả: `docs/eval_models_results.jsonl`.
- Dò alpha trộn (nhãn v4): alpha 0 → P@10 .147; 0.5 → .202; **0.85 → .227**; 1.0 → .228.
- Mô phỏng F2 chunk theo số chunk (nhãn v5): 10→.049, 30→.089, 60→.107 – vẫn tăng → nên 80–100 chunk.
- Doc: 120 tốt hơn 150. Thử 100 và 130.
- **Giả thuyết lớn:** BTC baseline có Docs Precision .88 (!) → gold nhiều khả năng được tạo từ BM25 của BTC rồi lọc.
  Index BM25 "chuẩn" đã build: `data/cache/bm25_vi_uni` (chạy với env `R2AI_BM25_VI_DIR=data/cache/bm25_vi_uni R2AI_VI_BIGRAMS=0 R2AI_NO_PLAN=1`),
  ứng viên: `data/cache/vi_uni_preds_300.jsonl` (chỉ trùng 41/100 với BM25 cũ). Điểm GreenNode cho nó: `data/cache/rr_gn_uni200.npy` (đang chấm).
  Nếu v7 > v4 (0.1193) → làm v8 = uni BM25 + GreenNode rerank; cũng thử HỢP NHẤT ứng viên của cả 2 BM25 trước khi rerank.

**Kết luận mới (04/10 tối):**
- BM25 với tokenizer XLM-R/bge-m3 (`data/cache/bm25_vi_xlmr`, env `R2AI_HF_TOK=data/models/bge-m3`): offline chỉ tìm 3.96 doc đúng-đã-biết/top100
  so với 6.69 của BM25 gốc → giả thuyết "BTC dùng tokenizer HF cho BM25" bị bác (không nộp).
- Gộp ứng viên BM25 gốc + BM25 chuẩn: chỉ +4% doc đúng-đã-biết (9.08→9.46) → lợi ích nhỏ.
- **Điểm nghẽn là RECALL ỨNG VIÊN**: gold ~71 doc/câu nhưng v8 chỉ trúng ~16 trong top120; phần lớn gold KHÔNG có trong top300 BM25.
  ⇒ cần nguồn ứng viên dense trên toàn bộ corpus VI: `scripts/dense_vi.py` (GreenNode, title+1500 ký tự đầu, 625k doc, checkpoint
  `data/cache/dense_gn/partNNN.npy`, ~25k doc/part). Sau đó `search` → top300 dense/câu → hợp nhất RRF với BM25+GreenNode-chunk → nộp.

**Quy trình v9 (dense) – đang chạy bước 1 lúc 21:30 ngày 04/10:**
```bash
nice .venv/bin/python scripts/dense_vi.py encode --out data/cache/dense_gn --max-len 256 --batch 64   # ~100 phút, 26 part, chạy lại sẽ bỏ qua part đã có
nice .venv/bin/python scripts/dense_vi.py search --index data/cache/dense_gn --k 300 --out data/cache/dense_gn_top300.json
for w in 0.5 1.0 2.0; do .venv/bin/python scripts/fuse_dense.py --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy \
    --dense data/cache/dense_gn_top300.json --w-dense $w --eval-only; done        # chọn w có nhiều doc đúng-đã-biết nhất
.venv/bin/python scripts/fuse_dense.py ... --w-dense <w tốt nhất> --n-docs 120 --n-chunks 80 --out submissions/v9_dense.zip   # rồi nộp
```
Lưu ý: "doc đúng-đã-biết" lấy từ các lần nộp trước (thiên về BM25) → dense có thể tốt hơn con số offline cho thấy; nên nộp ít nhất 1 bài w=1.0.

**KAGGLE (đã chạy 05/10 20:30):** token ở `~/.kaggle/access_token` (KHÔNG commit; dùng `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)`).
- Dataset riêng tư `kakaoffical/r2ai-vi-docs` = 628,732 doc VI gộp thành 9 parquet + query.parquet (tạo bằng script gộp shard, xem lịch sử chat/commit).
  Cập nhật khi crawl thêm: thêm file parquet mới vào thư mục rồi `kaggle datasets version -p <dir> -m "..."`.
- Kernel `kakaoffical/r2ai-dense-chunks` (thư mục `kaggle/`: `dense_chunks.py` + `kernel-metadata.json`, GPU, internet):
  dense CẤP ĐOẠN GreenNode (≤6 cửa sổ ~1600 ký tự/doc) → `dense_chunks_top.json` {qid: [[score, doc_id, chunk_text]...300 doc]}.
  Chạy: `kaggle kernels push -p kaggle`; theo dõi `kaggle kernels status kakaoffical/r2ai-dense-chunks`;
  lấy kết quả `kaggle kernels output kakaoffical/r2ai-dense-chunks -p data/cache/kaggle_out`.
  Sau đó gộp: nguồn RRF mới trong `fuse_dense.py` (doc từ dense_chunks dùng luôn chunk_text của nó).

**05/10 tối – tổng kết:** bài tốt nhất vẫn **v10 = 0.2097**. Đã thử, KHÔNG cải thiện: v12 thêm dense bge-m3 (0.2088), v13 chọn lại chunk
bằng GreenNode cho doc từ dense (0.2096), v14 100 doc (0.2081). Offline: ensemble GreenNode + bge-reranker-v2-m3 chỉ P@10 .232→.237 (+2%, chưa đáng chạy 240k cặp).
⇒ Đã chạm trần của "dense trên 1500 ký tự đầu + BM25". Đòn bẩy còn lại: (a) **dense cấp ĐOẠN** (`kaggle/dense_chunks.py`, cần Kaggle GPU – chờ token
`~/.kaggle/kaggle.json`), (b) **crawl nốt VI** – đang chạy: `logs/crawl_vi_rest.log` (110k URL, chủ yếu nhathuoclongchau 74k, phunusuckhoe 27k) →
xong thì `pipeline_v1.py build` lại + `dense_vi.py encode` (chỉ part mới) + `search` + `fuse_dense.py`.
**Lưu ý:** `configs/default.yaml` từng bị đặt `flush_every: 20` (sinh hàng nghìn shard nhỏ) – đã sửa về 5000.

**05/10:** v12 (thêm dense bge-m3 làm nguồn RRF thứ 3) = **0.2088** ≈ v10 0.2097 → bge-m3 không giúp (trùng GreenNode).
Chunk của doc CHỈ đến từ dense trúng 12.1% vs 16.9% của doc từ BM25 (đã chấm đoạn bằng GreenNode) →
`scripts/chunk_rerank.py` chọn lại chunk cho nhóm này bằng GreenNode (4 cửa sổ ~1600 ký tự/doc + chunk cũ) → v13 (đang chạy).

**(cũ) SẴN SÀNG NỘP (05/10, sau 07:00 – hạn mức ngày reset theo UTC = 07:00 giờ VN):**
- `submissions/v12_gn2_bgem3_1.zip`: RRF 3 nguồn = BM25+GreenNode-chunk ⊕ dense GreenNode (w=2) ⊕ dense **bge-m3** (w=1, `data/cache/dense_bgem3_top300.json`,
  encode xong toàn corpus `data/cache/dense_bgem3/`). Lệnh tạo:
  `fuse_dense.py --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy --dense data/cache/dense_gn_top300.json --w-dense 2.0 --dense2 data/cache/dense_bgem3_top300.json --w-dense2 1.0 --n-docs 120 --n-chunks 80`
  So với v10 (0.2097). Nếu tốt hơn → thử w-dense2 = 2.0; nếu kém → thử 0.5.

**KẾ HOẠCH TIẾP (từ 05/10, sau v10 = 0.2097):**
1. **Thêm model dense thứ 2** để ensemble trên toàn corpus: `BAAI/bge-m3` (đã có ở `data/models/bge-m3`) hoặc `Qwen/Qwen3-Embedding-0.6B`.
   Sửa `MODEL` trong `scripts/dense_vi.py` (hoặc thêm `--model`), encode ra thư mục khác (~100 phút), search, rồi RRF 3 nguồn
   (BM25+GN-chunk, dense GN, dense model 2) trong `fuse_dense.py`.
2. **Chunk cho doc từ dense**: hiện chọn bằng chunk_v2 (từ khoá). Nên chấm các đoạn ~1600 ký tự của doc đó bằng GreenNode và lấy đoạn tốt nhất
   (chunk F2 mới 0.154, doc F2 0.266 – chunk còn nhiều dư địa).
3. **Dense cấp đoạn** (encode nhiều đoạn/doc thay vì 1500 ký tự đầu) – tốn GPU hơn (~3–5×), nên làm trên Kaggle 2×T4.
4. Dò lại số doc/chunk khi đã có dense: thử 100 và 150 doc, 80 và 100 chunk (nhớ giới hạn ~56MB/zip).
5. Crawl nốt ~110k trang VI còn thiếu → encode thêm vào dense + rebuild BM25.

**Việc tiếp theo theo thứ tự (cũ):**
1. Đọc điểm v7 (ID mới nhất) → quyết định BM25 nào làm nguồn ứng viên (hoặc hợp nhất cả hai).
2. v8: rerank GreenNode trên nguồn tốt nhất, 120 doc + 80 chunk.
3. Ensemble model: cộng thứ hạng GreenNode + bge-reranker-v2-m3 (kiểm offline bằng eval_models trước).
4. Thử BM25 cấp CHUNK (đoạn ~1700 ký tự) thay vì cấp doc, rồi gộp lên doc bằng max – giống cách BTC có thể đã làm.
5. Crawl nốt ~110k trang VI còn thiếu (`python -m src.crawl --lang vi --per-domain 8`, máy local, `nice`), rebuild index.

**Lưu ý vận hành:**
- Upload zip lên leaderboard đôi khi **kẹt ở ~83–88%** không báo lỗi → reload trang và nộp lại (lượt nộp kẹt KHÔNG bị tính).
  File > ~60MB dễ kẹt hơn; 100 doc + 60 chunk ≈ 42MB, 150 doc + 80 chunk ≈ 56MB.
- `pkill -f "<pattern>"` trong cùng dòng lệnh sẽ tự giết shell của chính nó → dùng `pgrep` lấy PID rồi `kill <pid>`.
- GPU local (RTX 3050): luôn dùng fp16 (`rerank_vi.py` đã làm), ~50 đoạn/giây với GreenNode.

---

## 1. Những gì ĐÃ XÁC MINH về cách chấm (rất quan trọng)

Nguồn: trang Evaluation/Submission Instructions + `scoring_result.zip` (evaluation.json, details.jsonl) của các lần nộp.

| Điều | Chi tiết |
|---|---|
| Điểm | `FINAL = (DOCS_F2MACRO + CHUNKS_F2MACRO) / 2`. F2 = 5PR/(4P+R), tính từng câu rồi trung bình (macro). |
| Public test | Chấm **600/1200** câu (nhưng file nộp PHẢI đủ 1200 id; private chấm phần còn lại của cùng 1200 câu). |
| **doc_id** | **Phải là số nguyên** (= `id` trong `links_corpus.parquet`). Nếu là chuỗi `"123"` → scorer hiểu `pubmed:123` → 0 điểm. |
| PubMed | **Vô giá trị**: "mọi nhãn tài liệu đều dùng id trong links_corpus; tài liệu ngoài danh sách không được tính". Đừng làm nhánh tiếng Anh. |
| Số gold / câu | doc: trung vị **60**, TB 71, max 190 (7 câu = 0). chunk: TB ~75. ≈ 1 gold chunk / gold doc. |
| Khớp chunk | Chuẩn hoá NFKC + lowercase + gộp khoảng trắng; `overlap = LCS_tokens(pred, gold) / |gold|` ≥ **0.4**, cùng doc_id, cùng câu. Trùng ≥0.8 bị loại. Mẫu số là độ dài GOLD ⇒ chunk dài dễ phủ gold. |
| Độ dài gold chunk | Ước tính ~470 token ≈ **1600–1800 ký tự** (VI). 73% chunk trúng nằm trong 1500 ký tự đầu bài. |
| Ngôn ngữ của gold | Số doc VI trúng **bão hoà ~8/câu** dù gold 57→170 ⇒ **phần lớn gold là trang TIẾNG TRUNG** (corpus 83% là ZH). |
| Hạn | Hệ thống đóng 07:00 (GMT+7) – nộp trước 07:00. Public đến 31/10; private 01–04/11, **chỉ 5 bài tổng**. Tối đa 10 bài/ngày. |

### Lịch sử nộp (`experiments.csv`)
| ID | File | Cấu hình | FINAL | Doc P/R | Chunk P/R |
|---|---|---|---|---|---|
| 4186 | baseline_v0 | BM25 VI 205k doc, top3, doc_id **str** | 0.0000 | – | – |
| 4191 | baseline_v0_int | như trên, doc_id int | 0.0077 | .226/.013 | .004/.0001 |
| 4195 | v1_k40 | BM25 VI 625k doc, top40, chunk 800 ký tự | 0.0728 | .164/.133 | .032/.029 |
| 4208 | **v2_vi40_c1600x2** | pipeline_v2: VI top40 + `chunk_v2` (~1700 ký tự, ưu tiên đầu bài) | **0.0988** | .178/.141 | .092/.075 |

Chi tiết per-query của 4195 và 4208: `data/eval/details_4195_v1.jsonl`, `data/eval/details_4208_v2.jsonl` (không có trong git – xem mục 3).

---

## 2. Kiến trúc & file

```
src/crawl.py, src/extractors/   crawler async (curl_cffi giả Chrome, GBK/GB18030, resumable sqlite, extractor riêng cho top domain)
src/index.py                    tokenize(text, lang): vi = âm tiết + bigram; zh = jieba; en = stem
src/submit.py                   validate / complete / write_zip (validator dùng doc_id CHUỖI → chỉ chuyển INT khi ghi zip)
src/evaluate.py, src/select.py  metric offline + ngưỡng động (chưa dùng nhiều)
src/query.py, src/llm.py        Qwen3-4B-Instruct-2507 (4-bit) sinh QueryPlan: vi/en/zh rewrites + keywords → data/cache/query_plans.jsonl (đã có đủ 1200)
src/chunk.py, embed.py, retrieve.py, rerank.py   BGE-M3 + bge-reranker-v2-m3 (đã viết, CHƯA dùng trong bài nộp)
scripts/pipeline_v1.py          BM25 VI shard + hashed vocab (build ~5 phút, ~5GB RAM) → data/cache/bm25_vi/
scripts/pipeline_v2.py          predict VI (+ZH nếu có index) + chunk_v2; --dump-preds để xuất preds jsonl
scripts/chunk_v2.py             chọn chunk: cửa sổ ~1600–1800 ký tự quanh đoạn khớp, lead_bonus, answer_bonus, keywords từ plan
scripts/zh_index.py             BM25 ZH cục bộ (agent C viết, chưa dùng)
scripts/merge_preds.py          gộp VI preds + ZH preds theo quota --n-vi/--n-zh → zip nộp (doc_id int)
colab/zh_search_fast.py         BM25 ZH trên Colab: 1 lượt jieba → CSR băm → nhân ma trận với 1200 truy vấn → top-k + chunk
colab/zh_search.py              (bản chậm của agent; zh_search_fast import tokenize/select_chunks từ đây)
colab/setup_v3.sh, run_zh.sh    script dựng phiên Colab
docs/                           báo cáo các agent, thiết kế index, ghi chú PubMed (đã biết là vô dụng)
```

---

## 3. Dữ liệu (KHÔNG có trong git – quá lớn) và trạng thái

| Dữ liệu | Ở đâu | Cách tạo lại |
|---|---|---|
| Corpus URL + queries | HF `AIGuruTinix/ViBioMIR` → `data/ViBioMIR/` | `hf download AIGuruTinix/ViBioMIR --repo-type dataset --local-dir data/ViBioMIR` |
| Trang VI đã crawl (~627k/738k) | máy local `data/docs/corpus-*.parquet` (~700MB) | `python -m src.crawl --lang vi --per-domain 8 --delay 0.1` (resumable) |
| Index BM25 VI | `data/cache/bm25_vi/` (3.7GB) | `python scripts/pipeline_v1.py build` |
| Query plans 1200 câu | `data/cache/query_plans.jsonl` | `python -m src.query --queries data/ViBioMIR/query.parquet --out data/cache/query_plans.jsonl` (GPU, ~1.5h) |
| Trang ZH đã crawl (~800k) | Google Drive `MyDrive/r2ai/docs_zh/` + Colab `/content/docs_zh` (mất khi phiên Colab chết) | Colab: `colab/run_zh.sh` / restart (xem mục 5) |
| Kết quả tìm ZH | `zh_preds.jsonl` (Colab → tải về `~/Downloads`) | `python colab/zh_search_fast.py --docs /content/docs_zh --plans query_plans.jsonl --out /content/zh_preds.jsonl --k 100` |

Tình trạng crawl ZH (đến 04/10 ~02:00): bài viết (a-hospital, familydoctor, zhongyibaodian…) ~540k/976k; 120ask ~270k/951k;
cnkang ~19k/911k (**rất chậm ~1 trang/s**, có thể bị giới hạn); **39.net bị captcha** (>90% rỗng, cả máy local lẫn Colab);
**zysjonline (243k) bị Cloudflare**; iiyi.com (30k) server chết.

---

## 4. Quy trình tạo & nộp 1 bài

```bash
# VI preds (máy local, ~2 phút)
nice .venv/bin/python scripts/pipeline_v2.py predict --out submissions/vi_tmp.zip \
    --doc-max-vi 60 --doc-max-zh 0 --chunk-chars-vi 1600 --dump-preds data/cache/vi_preds_60.jsonl
# Gộp với ZH (zh_preds.jsonl từ Colab)
.venv/bin/python scripts/merge_preds.py --vi-preds data/cache/vi_preds_60.jsonl \
    --zh-preds ~/Downloads/zh_preds.jsonl --n-vi 20 --n-zh 60 --out submissions/v3_vi20_zh60.zip
# Kiểm tra: đủ 1200 id, doc_id int, 1 file json ở gốc zip
```
**Nộp**: https://leaderboard.aiguru.com.vn/competitions/15/ → tab "Bài nộp của tôi" → nút upload (Codabench).
Đăng nhập bằng tài khoản đội (file `.r2ai_cred` ở máy local, KHÔNG commit). Script nộp tự động bằng API chưa chạy được
(agent đã thử `scripts/submit_*.py` – lỗi CSRF/upload) → nộp qua trình duyệt (Chrome DevTools MCP) là cách đã chạy ổn.
**Đọc kết quả chi tiết** (khi đã đăng nhập, trong trình duyệt): `fetch('/api/submissions/<id>/get_details/')` → `scoring_result` (URL ký sẵn, curl tải được)
→ unzip → `scores.txt`, `evaluation.json`, `details.jsonl` (mỗi câu: gold count, relevant_predictions; mỗi chunk: doc_id `corpus:<id>`, relevant, best_overlap).
Ghi mọi lần nộp vào `experiments.csv`. **Mỗi lần nộp chỉ đổi 1 yếu tố.**

---

## 5. Colab – các lỗi đã gặp (đừng lặp lại)

1. **Drive FUSE không cho rename file tạm** → crawler ghi shard vào `/content/...` (local) rồi `cp -n` sang Drive mỗi 10 phút.
2. **Interrupt cell giết luôn tiến trình con** → luôn khởi động job dài bằng `setsid nohup ... < /dev/null &`.
3. **Colab ngắt khi không có cell nào chạy** (~90 phút) → luôn để 1 cell vòng lặp `while True: print(...); time.sleep(300)`.
4. **Giới hạn 12h/phiên** (bản free) → job phải xong trong 1 phiên hoặc checkpoint lên Drive.
5. **Drive có thể mất kết nối** ("Transport endpoint is not connected") → ghi kết quả quan trọng ra local trước, rồi `files.download(...)`.
6. Crawler chạy nhiều tiến trình trên Colab về sau flush shard rất nhỏ (20 doc) → hàng nghìn file; nên tăng `flush_every` hoặc gộp shard.
7. Gõ code vào ô Monaco bằng bàn phím bị autocomplete làm hỏng → đặt nội dung bằng `monaco.editor.getEditors()[i].setValue(code)`.

---

## 5b. Kaggle (nền tảng CHÍNH cho job nặng) – Colab là dự phòng khi hết quota Kaggle

Vì sao: Kaggle free có 4 vCPU / ~30GB RAM, GPU 2×T4 (hoặc P100) ~30h/tuần, **chạy nền bằng "Save & Run All (Commit)"
không cần mở trình duyệt** (tối đa 12h/lần), dữ liệu để trong **Kaggle Dataset** ổn định (không gặp các lỗi Drive ở mục 5).
Colab vẫn giữ nguyên các script (`colab/*.sh`, `colab/zh_search_fast.py`) – dùng khi Kaggle hết giờ GPU/tuần hoặc cần thử nhanh.

### Bước 1 – Một lần: cài Kaggle CLI + token
```bash
.venv/bin/pip install kaggle
# kaggle.com → Settings → API → "Create New Token" → lưu ~/.kaggle/kaggle.json ; chmod 600 ~/.kaggle/kaggle.json
# Bật Internet cho notebook cần xác minh số điện thoại tài khoản Kaggle.
```

### Bước 2 – Đưa dữ liệu lên Kaggle Dataset (riêng tư)
```bash
mkdir -p kaggle_ds/r2ai-data && cd kaggle_ds/r2ai-data
cp ../../data/cache/query_plans.jsonl .
# ZH: tải MyDrive/r2ai/docs_zh/*.parquet về (Drive desktop / rclone / tải zip từ web Drive) rồi GỘP shard nhỏ:
python - <<'PY'
import glob, pyarrow as pa, pyarrow.parquet as pq
fs = sorted(glob.glob('/path/to/docs_zh/corpus-*.parquet'))
for i in range(0, len(fs), 50):          # gộp 50 shard → 1 file để tránh hàng nghìn file nhỏ
    pq.write_table(pa.concat_tables([pq.read_table(f) for f in fs[i:i+50]]), f'zh-{i//50:04d}.parquet', compression='zstd')
PY
cp ../../data/docs/corpus-*.parquet .     # VI (~700MB) – đổi tên vi-*.parquet nếu muốn
kaggle datasets init -p .                 # sửa dataset-metadata.json: "title": "r2ai-data", "id": "<user>/r2ai-data"
kaggle datasets create -p . --dir-mode zip          # lần đầu (mặc định private)
kaggle datasets version -p . -m "thêm shard ZH" --dir-mode zip   # các lần sau
```
Code: tạo thêm dataset `<user>/r2ai-code` từ repo này (hoặc trong notebook `!git clone https://github.com/kaka2k1/ai-rag`).

### Bước 3 – Notebook Kaggle chạy pipeline (commit chạy nền)
Tạo `kaggle/run_pipeline.ipynb` (hoặc script) với các ô:
```python
!git clone -q https://github.com/kaka2k1/ai-rag /kaggle/working/ai-rag
%cd /kaggle/working/ai-rag
!pip -q install jieba bm25s pyvi
DATA = '/kaggle/input/r2ai-data'
# 1) ZH BM25 – dùng 4 nhân: thêm multiprocessing.Pool(4) cho bước jieba trong colab/zh_search_fast.py
!python colab/zh_search_fast.py --docs {DATA} --plans {DATA}/query_plans.jsonl --out /kaggle/working/zh_preds.jsonl --k 200
# 2) VI BM25: pipeline_v1 build + pipeline_v2 predict --dump-preds (đặt data/docs -> symlink tới {DATA})
# 3) (GPU) rerank: src/rerank.py (bge-reranker-v2-m3) trên top-200 mỗi ngôn ngữ → cắt ngưỡng
# 4) scripts/merge_preds.py → /kaggle/working/submission.zip
```
Chạy: nút **Save Version → Save & Run All (Commit)**, chọn Accelerator GPU T4×2 khi có bước rerank, bật Internet.
Hoặc từ CLI: `kaggle kernels push -p kaggle/` (cần `kernel-metadata.json` với `"dataset_sources": ["<user>/r2ai-data"]`,
`"enable_gpu": true`, `"enable_internet": true`), xem trạng thái `kaggle kernels status <user>/<slug>`,
lấy kết quả `kaggle kernels output <user>/<slug> -p out/` → nộp `out/submission.zip`.

### Lưu ý
- `zh_search_fast.py` hiện tách từ 1 luồng (~17s/shard 5k doc trên Colab) → trên Kaggle nên song song hoá jieba (Pool(4)) ≈ 2–3× nhanh hơn.
- Crawl dài (>12h) KHÔNG hợp Kaggle/Colab → crawl trên máy local (`nice -n 15`), định kỳ `kaggle datasets version` để đẩy shard mới.
- Hết quota GPU Kaggle (30h/tuần) → quay về Colab: `colab/setup_v3.sh` + mục 5 (nhớ các lỗi đã gặp).

## 5c. agy có điều khiển được trình duyệt không? (đã kiểm tra 04/10)
- `agy -p` mặc định **KHÔNG** có tool trình duyệt (chỉ `read_url_content` đọc HTML tĩnh). Chưa có MCP nào (`agy mcp list`).
- Bật được bằng Chrome DevTools MCP. **Khuyến nghị dùng trình duyệt cô lập** (không đụng Chrome chính đang đăng nhập email/công việc):
  `agy mcp add chrome-devtools npx -y chrome-devtools-mcp@latest --isolated`
  Sau đó agy tự đăng nhập leaderboard bằng `.r2ai_cred` và nộp qua UI. (`--autoConnect` = điều khiển Chrome chính của bạn – rủi ro cao hơn.)
- Thay thế không cần MCP: nộp bằng script Playwright (`pip install playwright && playwright install chromium`) – agy viết & chạy qua run_command.

---

## 6. Lộ trình tối ưu (ưu tiên từ trên xuống)

> **CẬP NHẬT 04/10 (quan trọng hơn mọi mục bên dưới):** v3 gộp ZH (BM25 jieba trên 780k trang ZH) **làm GIẢM điểm**
> (0.0553 / 0.0739; chunk ZH trúng chỉ 0.3%). **v4 = chỉ VI, top100 doc = 0.1193 (tốt nhất)**, 12 doc trúng/câu,
> tỉ lệ trúng giảm rất chậm theo rank (10% → 6.5%) ⇒ gold chủ yếu là trang **VI** nằm sâu. Ưu tiên mới:
> 1. Xếp hạng lại VI bằng model ngữ nghĩa. Đánh giá offline KHÔNG tốn lượt nộp: `scripts/eval_models.py`
>    (nhãn thật từ `data/eval/details_4286_v4.jsonl`). Baseline thứ tự BM25: **AUC 0.559, P@10 0.116** – gần ngẫu nhiên → nhiều dư địa.
>    Ứng viên (mở, hợp lệ): `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1`, `BAAI/bge-m3`, `Qwen/Qwen3-Embedding-0.6B|4B`,
>    `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen3-Reranker-0.6B|4B`. KHÔNG dùng `text-embedding-004` (Gemini – đóng) hay API nội bộ công ty.
> 2. Tăng số doc VI: thử 150/200 (precision giảm chậm, F2 ưu tiên recall); retrieve top-300 BM25 rồi rerank, cắt ~100–150.
> 3. Dense retrieval VI (embedding toàn bộ ~625k doc VI trên Kaggle GPU) để lấy ứng viên mà BM25 bỏ sót, hợp nhất RRF.
> 4. Crawl nốt ~110k trang VI còn thiếu. ZH để sau (chỉ thử lại khi có retrieval xuyên ngữ bằng embedding).

1. **Nộp v3 = VI + ZH** (bước kỳ vọng tăng nhiều nhất). Thử quota: (20 VI + 60 ZH), (30 VI + 40 ZH), (10 VI + 90 ZH).
   Đọc details.jsonl: tỉ lệ doc ZH trúng theo thứ hạng → chọn số doc ZH tối ưu cho F2.
2. **Số doc/câu tối ưu cho F2**: gold trung vị 60 → thường trả 60–100 doc tốt hơn 40. Dùng details.jsonl (precision theo rank) để mô phỏng F2 offline cho các mức cắt.
3. **Truy vấn ZH tốt hơn**: plan["zh"] từ Qwen3-4B có lỗi thuật ngữ; thử Qwen3-8B / Qwen2.5-7B (≤15B, trước 08/2026) trên Colab GPU,
   hoặc dịch thêm nhiều biến thể; thêm từ khoá triệu chứng/bệnh tiếng Trung chuẩn (医学术语).
4. **Rerank / dense trên Colab GPU (T4)**: BGE-M3 dense cho top-200 ứng viên BM25 mỗi ngôn ngữ, rồi `bge-reranker-v2-m3` (query VI ↔ doc VI/ZH) để sắp lại và cắt ngưỡng.
   Code có sẵn `src/embed.py`, `src/rerank.py`. Hợp lệ (mở, <15B, phát hành 2024).
5. **Chunk**: gold ≈1600–1800 ký tự VI; ZH thử 600/900/1200 ký tự; thử 2 chunk/doc (lead + đoạn khớp nhất) – precision chunk tính theo chunk nên cân nhắc.
6. **Crawl thêm ZH**: phần còn lại của bài viết & 120ask; cnkang (thử giảm concurrency / đổi delay); 39.net captcha – thử `curl_cffi` impersonate khác hoặc bỏ.
7. Cân bằng ngôn ngữ theo câu: câu có nhiều doc VI điểm cao → tăng VI; ngược lại tăng ZH.
8. Viết **working notes** (bắt buộc để kết quả được công nhận): mô hình + ngày phát hành, nguồn dữ liệu, ablation từ experiments.csv.

---

## 7. Dùng agy (Antigravity CLI) để làm tiếp

```bash
cd ~/Desktop/hsk/test2      # (hoặc thư mục clone repo này + dữ liệu ở mục 3)
agy -p "Đọc HANDOFF.md rồi thực hiện mục 6.1: tạo các bài v3, ghi experiments.csv. Không commit .r2ai_cred." \
    --model gemini-3.1-pro-high --dangerously-skip-permissions --add-dir "$PWD"
```
- Model: `gemini-3.1-pro-high` (quota Gemini riêng), `claude-opus-4-6-thinking` (quota Claude, hết nhanh), `gemini-3.8-flash-high` (rẻ).
- Chỉ **1 agent được nộp bài** để không vượt 10 lượt/ngày; các agent khác chỉ tạo zip + phân tích.
- Máy local yếu (RTX 3050 8GB, 31GB RAM): mọi job nặng `nice -n 15`, RAM ≤ 12GB; model GPU nặng → Colab.
- Seat agy khác (`agyseat2/3`) không đọc được `/home/phong` (quyền 750) → muốn dùng phải đặt dự án ở thư mục chung (vd `/srv/r2ai`, group chung) và gọi `sudo -n -u agyseat2 -H -- /usr/local/bin/agy -p ...`.

## 8. Bảo mật
- **Không commit** `.r2ai_cred`, `logs/` (log agent có thể chứa mật khẩu), `test_login*.py`.
- Mật khẩu tài khoản thi đã lộ trong lịch sử chat → nên đổi.
