# ViBioMIR Live – ứng dụng tra cứu bằng chứng y khoa (chạy trực tiếp)

Nhận **câu hỏi tiếng Việt bất kỳ**, trả về tài liệu và đoạn bằng chứng trích nguyên văn từ ~738k trang tiếng Việt và ~222k trang tiếng Trung. Thời gian ~2,4 giây/câu trên RTX 3050 8 GB.

## Chạy
```bash
bash app_live/run.sh          # kiểm tra dữ liệu/cổng, nạp mô hình (~30 giây), tự mở trình duyệt khi sẵn sàng
# PORT=8800 bash app_live/run.sh · NO_BROWSER=1 bash app_live/run.sh · Ctrl+C để tắt · log: app_live/server.log
```
API: `POST /api/search {"q"}` (JSON) · `POST /api/search/stream {"q"}` (NDJSON, tiến trình thật từng bước) ·
`POST /api/translate {"text"}` (dịch ZH→VI, stream) · `GET /api/doc?lang=vi|zh&id=` · `GET /api/health`

## Giao diện
- Tiến trình xử lý **thật** từ máy chủ: thời gian từng bước, câu dịch tiếng Trung hiện ngay khi có.
- Toàn bộ n tài liệu hệ thống chọn (phân trang 10), lọc Việt/Trung, mức liên quan từ cross-encoder.
- Đoạn bằng chứng trích nguyên văn, tô từ khóa; **Xem trong tài liệu gốc** mở văn bản đầy đủ và cuộn tới đúng đoạn trích.
- Đoạn tiếng Trung: nút **Dịch sang tiếng Việt** (Qwen3-4B, hiện dần từng chữ, có chặn vòng lặp sinh).
- Tải kết quả JSON theo định dạng nộp bài (relevant_docs, relevant_chunks); sao chép trích dẫn kèm nguồn.
- Lịch sử tra cứu, gợi ý, phím tắt `/` (tìm) và `Esc`, nút Back của trình duyệt, giao diện điện thoại.

## Pipeline (server.py)
1. Ứng viên tiếng Việt: GreenNode dense (embedding tài liệu có sẵn `data/cache/dense_gn`, tìm chính xác theo khối trên GPU) ∪ FTS5 theo tiêu đề (`docs.sqlite`).
2. Dịch câu hỏi VI→ZH bằng Qwen3-4B-Instruct (4-bit) → bge-m3 dense trên `data/cache/zhfeat3_out/zh_emb.npy` (câu gốc + câu dịch).
3. Chấm lại bằng bge-reranker-v2-m3 trên cửa sổ đoạn khớp nhất; xác suất = sigmoid(logit − 3); ZH × 0.9.
4. Số tài liệu n = argmax F2 kỳ vọng 5·Σp₁..ₙ / (4·c·Σp + n), c = 1.9, n ≥ 10.

Khác bản thi: chưa dùng LightGBM ~60 đặc trưng, cross-encoder fine-tune và BM25 toàn văn (để vừa 1 GPU 8 GB và trả lời trong vài giây).

## Dữ liệu cần có
- `app_live/docs.sqlite` – kho văn bản (tạo bằng `python3 app_live/build_store.py`, rồi lệnh FTS5 rebuild trong lịch sử).
- `data/cache/dense_gn/`, `data/cache/zhfeat3_out/`, `data/models/bge-m3-st/`; các mô hình HF trong cache.
