# Chuẩn bị hỏi đáp – Chung kết R2AI 2026 (ViBioMIR · aichickenteam)

Hình thức: 5 phút trình bày + 5 phút hỏi đáp. Mỗi câu trả lời nên gói trong **20–40 giây**: trả lời thẳng ý chính trước, rồi một con số, rồi dừng. Giám khảo muốn hỏi sâu sẽ hỏi tiếp.

## Ba thông điệp cần nhắc lại khi có dịp

1. **Xuyên ngôn ngữ thật sự:** câu hỏi tiếng Việt được dịch sang tiếng Trung để tìm thêm bằng chứng. Nhánh tiếng Trung nâng điểm public từ 0.4095 lên 0.4488 (+0.039).
2. **Biết dừng đúng chỗ:** mỗi câu chọn số tài liệu tối đa F2 kỳ vọng (30–300 tài liệu) thay vì top-k cố định.
3. **Trung thực, tái hiện được:** đánh giá out-of-fold theo câu, khai báo đầy đủ nguồn nhãn, toàn bộ mã nguồn và dữ liệu đã gửi BTC.

---

## A. Kết quả

**A1. Điểm của đội là bao nhiêu?**
Vòng private đạt **0.3801**. Bốn bài nộp private với cấu hình khác nhau nằm trong khoảng 0.3778–0.3801, chênh dưới 0.003, nên hệ thống ổn định. Ở public, hệ thống đạt 0.4488 theo luật chấm cũ và 0.4077 theo luật chấm đoạn mới (BTC đổi ngày 23/10).
*(Không nhắc thứ hạng.)*

**A2. Vì sao điểm private (0.3801) thấp hơn public (0.4077)?**
Hai tập có câu hỏi khác nhau nên chênh lệch là bình thường. Đội đã hạn chế khớp quá mức vào public: mọi xác suất của câu public đều là out-of-fold, tức mô hình chấm một câu không thấy nhãn của chính câu đó. Một số đặc trưng như "tài liệu đúng của câu hỏi tương tự" và tiên nghiệm tên miền có lợi hơn cho câu public vì chúng cùng tập với nhãn. Đây là hướng đội muốn cải thiện.
*(Nói trung thực, không đoán thêm nguyên nhân khác.)*

**A3. Luật chấm đoạn mới ảnh hưởng thế nào?**
Luật mới yêu cầu độ trùng LCS ≥ 40% so với **cả** độ dài đoạn đúng **và** độ dài đoạn dự đoán. Đoạn dài không còn được lợi, nên điểm cùng hệ thống giảm từ 0.449 xuống khoảng 0.408. Đội đã quét lại độ dài đoạn (3.000–8.000 ký tự, vị trí cắt, đoạn đôi, đoạn ngắn) và chọn đoạn tiếng Việt ≤ 5.000 ký tự, tiếng Trung 1.800 ký tự. Độ dài gần như không đổi điểm, còn đoạn quá ngắn (1.200 ký tự) thì tệ hẳn.

**A4. Đóng góp lớn nhất đến từ đâu?**
Theo thứ tự trên public: LightGBM xếp hạng kết hợp cross-encoder fine-tune (0.298 → 0.408), nhánh tiếng Trung (→ 0.435), dịch câu hỏi sang tiếng Trung (→ 0.449). Huấn luyện lại cross-encoder trên toàn bộ nhãn tích lũy cũng là một đòn bẩy lớn.

---

## B. Phương pháp

**B1. Tóm tắt hệ thống trong một câu?**
Truy hồi nhiều nguồn (BM25 + mô hình nhúng) → xếp hạng lại bằng LightGBM trên khoảng 60 đặc trưng → trộn tiếng Việt và tiếng Trung theo xác suất → chọn số tài liệu tối ưu F2 → trích đoạn nguyên văn.

**B2. Chọn số tài liệu tối ưu F2 như thế nào?**
LightGBM cho mỗi tài liệu một xác suất liên quan p. Với n tài liệu đầu, F2 kỳ vọng ≈ 5·Σp₁..ₙ / (4·c·Σp + n). Tử số là số tài liệu đúng kỳ vọng. c·Σp ước lượng tổng số tài liệu đúng, kể cả những tài liệu hệ thống chưa tìm thấy, vì vậy c = 1.9 > 1. Chọn n làm biểu thức lớn nhất, nên câu dễ trả về ít, câu khó/rộng trả về nhiều (30–300 tài liệu).

**B3. Vì sao dùng LightGBM mà không chỉ dùng cross-encoder?**
Cross-encoder chỉ là một tín hiệu. LightGBM kết hợp khoảng 60 tín hiệu (BM25, các điểm dense, cross-encoder zero-shot và fine-tune, độ tương đồng với nhóm tài liệu đầu, tiên nghiệm tên miền, câu tương tự) và cho ra **xác suất đã hiệu chỉnh**. Bước chọn số tài liệu F2 và bước trộn Việt–Trung đều cần xác suất này.

**B4. Dịch câu hỏi dùng mô hình gì? Có kiểm soát chất lượng không?**
Bài thi dùng Qwen2.5-7B-Instruct, giải mã greedy, prompt yêu cầu giữ nguyên thuật ngữ y khoa và chỉ xuất bản dịch. Bản dịch là một nguồn ứng viên chứ không thay câu gốc: nhánh tiếng Trung tìm bằng **cả** câu Việt gốc (bge-m3 đa ngôn ngữ) **và** câu đã dịch (bge-m3 Trung–Trung, BM25 jieba). Nhờ vậy dịch sai một từ thì vẫn còn các nguồn khác bù.
*(Ứng dụng demo dùng Qwen3-4B 4-bit cho nhẹ.)*

**B5. Trộn tài liệu Việt và Trung thế nào để không lệch?**
Mỗi nhánh có LightGBM riêng nhưng cùng cho xác suất đã hiệu chỉnh, nên đặt chung một thang. Xác suất tiếng Trung nhân 0.9 cho thận trọng. Trung bình khoảng 22–27 tài liệu tiếng Trung mỗi câu được chọn.

**B6. Cắt đoạn (chunking) thế nào?**
Cửa sổ khoảng 1.600 ký tự theo ranh giới câu, tối đa 6 cửa sổ mỗi tài liệu, để tính dense cấp đoạn. Đoạn nộp là cửa sổ khớp ngữ nghĩa nhất, nới tối đa 5.000 ký tự (tiếng Việt). Tiếng Trung lấy 1.800 ký tự đầu. Mọi đoạn đều trích nguyên văn.

**B7. Các mô hình có đúng quy định không?**
Có. Tất cả là mô hình mã nguồn mở, ≤ 15B tham số, phát hành trước 01/08/2026: GreenNode-Embedding-Large-VN, bge-m3, multilingual-e5-large-instruct, bge-reranker-v2-m3, Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507, cùng LightGBM. Không dùng LLM đóng hay API trả phí.

**B8. Vì sao không dùng LLM để sinh câu trả lời (RAG)?**
Đề yêu cầu trích **nguyên văn**, không sinh mới. Trong y khoa, trích nguồn gốc an toàn hơn để người dùng tự kiểm chứng. Đội đã thử Qwen3-8B làm "giám khảo" chấm liên quan: AUC 0.66, kém xa GreenNode (0.79), nên bỏ.

---

## C. Dữ liệu

**C1. Dữ liệu lấy từ đâu?**
BTC chỉ cung cấp danh sách URL. Đội tự thu thập **735.361 trang tiếng Việt** và **222.297 trang tiếng Trung** (94% bộ 235.518 URL tiếng Trung BTC chỉ định), tổng 957.658 trang. Không dùng dữ liệu y khoa ngoài.

**C2. Khó khăn khi thu thập?**
Một số trang Trung chặn IP Việt Nam (120ask, 39.net) nên phải cào song song từ nhiều kernel Kaggle. 39.net đi qua bản mobile. Pharmacity lấy lại qua tên miền mới theo hướng dẫn BTC, giữ nguyên doc_id. Văn bản mã GBK/GB18030 được xử lý riêng, và có bộ trích nội dung riêng cho các trang lớn.

**C3. Sao không dùng tiếng Anh (PubMed)?**
Đội đã thu thập 239.770 abstract PubMed và thử hai lần (theo doc_id nội bộ và theo PMID): cả hai cho **0 kết quả đúng** trên public, nên loại để không làm giảm precision. Nếu có nhãn tiếng Anh, pipeline thêm được ngay theo cách làm nhánh tiếng Trung.

**C4. Bản quyền/đạo đức khi thu thập?**
Chỉ thu thập đúng danh sách URL BTC cung cấp, cào lịch sự theo tên miền (giới hạn tốc độ, có độ trễ), dùng cho nghiên cứu của cuộc thi. Đầu ra luôn kèm đường dẫn nguồn.

---

## D. Nhãn và tính minh bạch (câu dễ bị hỏi khó nhất)

**D1. Đội lấy nhãn huấn luyện ở đâu khi BTC không cho tập huấn luyện?**
Từ tệp `details.jsonl` hệ thống chấm trả về sau mỗi lần nộp public: tệp cho biết đoạn nào được tính là khớp. Tổng cộng khoảng 363 nghìn cặp có nhãn, khoảng 32 nghìn cặp dương. Đội có nộp một số bài thăm dò để lấy thêm nhãn. Tất cả đã **khai báo đầy đủ trong tài liệu thuyết minh** theo thông báo của BTC ngày 23/10.

**D2. Như vậy có phải "học thuộc đáp án" public không?**
Không. Đội **không chép đáp án** vào bài nộp. Mọi xác suất cho câu public là **out-of-fold** (5 fold theo câu), nên điểm public phản ánh khả năng tổng quát hóa. Câu private không có nhãn nào và được chấm bằng mô hình huấn luyện trên nhãn public. Từ 22/10, khi BTC tắt thông tin chi tiết, đội không thu thêm nhãn. Lịch sử có trong `experiments.csv` và git log.

---

## E. Ứng dụng demo

**E1. Ứng dụng demo có phải hệ thống đã nộp không?**
Là **bản rút gọn** để chạy trực tiếp trên 1 GPU 8 GB trong vài giây: dense GreenNode + tìm theo tiêu đề cho tiếng Việt, dịch bằng Qwen3-4B, bge-m3 cho tiếng Trung, chấm lại bằng bge-reranker-v2-m3 và cùng công thức chọn số tài liệu F2. Bản rút gọn chưa có LightGBM 60 đặc trưng, cross-encoder fine-tune và BM25 toàn văn. **Điểm thi đến từ hệ thống đầy đủ.**

**E2. Tốc độ, phần cứng?**
Khoảng 2,5 giây mỗi câu trên RTX 3050 8 GB. Khởi động khoảng 30 giây. Tìm kiếm là tìm chính xác (không xấp xỉ) trên 738 nghìn + 222 nghìn vector, chia khối trên GPU.

**E3. Người dùng Việt đọc tài liệu tiếng Trung thế nào?**
Đoạn bằng chứng tiếng Trung được trích nguyên văn, có nút dịch sang tiếng Việt (hiện dần từng chữ, khoảng 5–9 giây). Bản dịch ghi rõ "bản dịch máy · tham khảo", còn nguồn gốc luôn mở được.

**E4. Làm sao biết đoạn trích không bị bịa?**
Nút "Xem trong tài liệu gốc" mở toàn văn và cuộn tới đúng đoạn trích đã tô sáng, kèm đường dẫn nguồn gốc. Hệ thống không sinh nội dung y khoa.

**E5. Ai sẽ dùng? Khác gì Google?**
Nhân viên y tế, dược sĩ nhà thuốc, người dân cần tra cứu có nguồn. Khác Google ở ba điểm: (1) tự tìm thêm nguồn tiếng Trung từ câu hỏi tiếng Việt; (2) trả về đoạn bằng chứng cụ thể, không chỉ đường dẫn; (3) chỉ tìm trong kho y khoa đã chọn lọc.

---

## F. Hạn chế và hướng phát triển

**F1. Hạn chế lớn nhất?**
Trần điểm do dữ liệu quyết định: còn khoảng 6% URL tiếng Trung chưa thu thập được, và chưa có nguồn tiếng Anh hữu ích. Cắt đoạn vẫn theo cửa sổ cố định, chưa theo ranh giới ngữ nghĩa.

**F2. Nếu có thêm thời gian?**
Cắt đoạn theo ngữ nghĩa, thêm reranker riêng cho cấp đoạn, và đóng gói ứng dụng tra cứu cho nhân viên y tế và nhà thuốc (bản demo hôm nay là bước đầu).

**F3. Mở rộng/triển khai thực tế cần gì?**
Kho tài liệu cập nhật định kỳ (crawler chạy tiếp được), chỉ mục có thể chuyển sang tìm xấp xỉ (FAISS) khi kho lớn hơn, và cần chuyên gia y tế đánh giá chất lượng bằng chứng trước khi dùng cho bệnh nhân.

---

## G. Câu hỏi bất ngờ – cách xử lý

- **Không biết câu trả lời:** "Phần này đội chưa đo chính xác; theo hiểu biết của đội thì … ; đội xin bổ sung sau qua email BTC." Không bịa số.
- **Bị chê điểm thấp hơn đội khác:** nhắc ba điểm khác biệt (xuyên ngôn ngữ, chọn số tài liệu theo F2, minh bạch/tái hiện được) và ứng dụng chạy thật.
- **Hỏi chi phí:** toàn bộ huấn luyện dùng GPU miễn phí (Kaggle T4×2, Colab) và một GPU cá nhân RTX 3050; không tốn chi phí API.
- **Hỏi thời gian chạy toàn bộ 1.200 câu:** `[điền: thời gian chạy pipeline đầy đủ, ví dụ từ log tạo bài nộp]`.
- **Hỏi phân công đội:** `[điền]`.

## Kiểm tra trước giờ thi (sáng 14/11)

- [ ] Khởi động `bash app_live/run.sh` lúc khoảng 08h10, chờ "Sẵn sàng", chạy thử 1 câu.
- [ ] Tắt các chương trình khác dùng GPU; thử chia sẻ cửa sổ trình duyệt trên Teams.
- [ ] Trình duyệt phóng to 125%, mở sẵn tab slide và tab ứng dụng.
- [ ] Video dự phòng `app_live/demo_backup.mp4` mở sẵn trong trình phát.
- [ ] Tên tài khoản Teams: "Tên đội – Họ tên trưởng nhóm".
