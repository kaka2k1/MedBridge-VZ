# HANDOFF – R2AI2026 Truy hồi tài liệu y khoa đa ngôn ngữ (bàn giao cho agy / người tiếp theo)

> Cập nhật: 04/10/2026. Đọc file này TRƯỚC khi làm gì. Mục tiêu: vượt **0.3764** (baseline BM25 của BTC, đang đứng đầu).
> Đề bài đầy đủ: `R2AI2026_cuoc_thi.md` (lưu ý quy định mô hình mới nhất trên trang Overview: phát hành **trước 01/08/2026**, **≤ 15B** tham số, cấm LLM đóng).

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

## 6. Lộ trình tối ưu (ưu tiên từ trên xuống)

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
