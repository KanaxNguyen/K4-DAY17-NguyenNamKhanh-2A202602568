# BƯỚC 8 + 9: PHÂN TÍCH KẾT QUẢ BENCHMARK VÀ BONUS
**Khóa học:** Phase 2, Track 3, Day 17: Memory Systems for AI Agent
**Họ và tên:** Nguyễn Nam Khánh — **Mã sinh viên:** 2A202602568

Mọi số liệu dưới đây sinh ra bởi `python src/benchmark.py` (offline, tất cả deterministic) và được lưu ở [`results/benchmark.md`](results/benchmark.md). Mỗi lần chạy dùng thư mục `state/` tạm **rỗng**, nên `Memory growth` là mức tăng thật từ 0 và không bị lẫn `User.md` của lần chạy trước. Kiểm chứng hành vi: `pytest src/test_agents.py -v` (18 test).

---

## 1. Kết quả

### Bảng 1: Standard Benchmark (10 hội thoại, 10 lượt, có correction và nhiễu)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Agent | 2,132 | 16,889 | 2.4% | 16.7% | 0 B | 0 |
| Advanced Agent | 1,289 | 21,944 | **100.0%** | 99.3% | 736 B | 0 |

### Bảng 2: Long-Context Stress Benchmark (1 hội thoại, 16 lượt rất dài)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Agent | 1,284 | 27,721 | 0.0% | 15.0% | 0 B | 0 |
| Advanced Agent | 322 | **8,138** (-71%) | **100.0%** | 100.0% | 487 B | 7 |

Baseline đạt 2.4% ở Standard vì câu hỏi conv-05 chứa sẵn chữ "DũngCT" và baseline lặp lại câu hỏi. Đây là trùng lặp ngẫu nhiên, không phải trí nhớ. Chuỗi viết tắt như `AI` được so khớp phân biệt hoa/thường để đại từ "ai" không bị tính là đúng.

### Bảng 3: Điểm hòa vốn của compact (length sweep, cùng một hội thoại cắt theo số lượt)
Hội thoại ngắn (conv-01, 1→10 lượt): Advanced luôn đắt hơn Baseline (+162% ở lượt 1, giảm dần còn +10% ở lượt 10), compact không bao giờ kích hoạt.
Hội thoại dài (stress-01): Baseline rẻ hơn ở 1–2 lượt (+24%, +17%). **Từ lượt 4 Advanced bắt đầu rẻ hơn** (-10%) và khoảng cách tăng dần: -30% (6 lượt), -43% (8), -62% (12), **-71% (16)**. Bảng đầy đủ nằm trong `results/benchmark.md`.

---

## 2. Bốn câu hỏi trọng tâm

### 2.1. Vì sao Advanced có recall tốt hơn Baseline?
Baseline chỉ giữ `messages` theo `thread_id`. Recall chạy ở thread mới nên baseline nhận thread rỗng và phải trả lời "chưa có thông tin". Advanced ghi fact ổn định vào `state/profiles/<user>/User.md` (persistent) và nạp lại ở mọi thread. Ba lớp memory tách bạch:

| Lớp | Nằm ở đâu | Sống bao lâu | Vai trò |
|---|---|---|---|
| Short-term | `CompactMemoryManager.messages` (các tin gần nhất) | trong 1 thread | mạch hội thoại hiện tại |
| Compact | `summary` + `open_threads` của thread | trong 1 thread | thay phần lịch sử cũ bằng bản tóm tắt chặn kích thước |
| Persistent | `User.md` | qua mọi thread | fact ổn định về người dùng |

Câu trả lời của Advanced chỉ lấy từ `User.md`, không có giá trị mặc định nào. Bản trước có lỗi: `_offline_response` có fallback cứng "DũngCT / Huế / MLOps engineer", nên recall 100% là nhờ đáp án viết sẵn. Bản này bỏ fallback. Khi thiếu fact, agent nói "chưa có thông tin trong User.md" (có test `test_advanced_does_not_fabricate_unknown_facts`).

### 2.2. Vì sao Advanced tốn token hơn ở hội thoại ngắn?
Standard: 21,944 so với 16,889 prompt tokens (+30%). Mỗi lượt Advanced phải nạp `User.md` trước, còn Baseline chỉ nạp lịch sử thread. Ở hội thoại ngắn tổng ngữ cảnh chưa vượt ngưỡng 600 token nên compact chưa kích hoạt (0 compaction) và `User.md` là chi phí cố định không có đối ứng. Bảng sweep cho thấy phần này đã nhỏ lại khi hội thoại dài hơn (+162% → +10%). `Agent tokens only` của Advanced thấp hơn vì câu ack ngắn, nhưng đó không phải lợi thế của compact. Compact **không** làm giảm token sinh ra.

### 2.3. Vì sao compact thắng ở hội thoại dài, và vì sao nó chủ yếu tối ưu `prompt tokens processed`?
Baseline gửi lại toàn bộ lịch sử ở mỗi lượt, nên tổng prompt tokens tăng gần bậc hai theo số lượt (stress: 27,721). Advanced giữ K=4 tin gần nhất cộng một bản tóm tắt chặn tối đa 4 chủ đề và 3 "việc còn mở", nên mỗi lượt có kích thước gần như không đổi, tổng tăng tuyến tính (8,138, 7 compaction). Compact chỉ thay đổi **phần ngữ cảnh được gửi vào mỗi lượt**. Số token model sinh ra (`Agent tokens only`) và chất lượng recall phụ thuộc vào model và `User.md`, không phụ thuộc compact. Compact còn có chi phí: nó làm mất chi tiết cũ (mức độ hoặc con số trong tin tức), chỉ giữ chủ đề. Vì vậy mọi fact cần nhớ lâu phải đi vào `User.md`, chứ không dựa vào summary.

### 2.4. File memory phình ra thế nào và rủi ro gì?
`User.md` tăng 736 B (Standard) và 487 B (Stress). Con số nhỏ vì mỗi key chỉ có đúng một dòng, nên file tăng theo số **key**, không theo số **lượt chat**. Rủi ro thực tế:
1. **Phình to / ngữ cảnh nền tăng**: nếu cho phép nhiều key tự do, User.md sẽ lớn dần và bị nạp vào *mọi* lượt. Đã chặn bằng: một dòng một key, field dạng danh sách giới hạn 5 mục, decay loại fact cũ khỏi prompt.
2. **Lưu sai fact**: câu đùa, ví dụ, chuyến công tác bị ghi thành fact vĩnh viễn (xem bonus A).
3. **Fact mâu thuẫn**: giữ cả "ở Đà Nẵng" lẫn "ở Huế" (xem bonus B).
4. **Quyền riêng tư**: `User.md` là plain text, chứa PII. Thực tế cần mã hóa, phân quyền và cho người dùng quyền xem/xóa. Repo này chưa làm phần đó.

---

## 3. Bonus (Bước 9): 4 cơ chế, mỗi cơ chế giải quyết gì, cải thiện gì, rủi ro gì

Cả bốn đều nằm trong `memory_store.py`, và `User.md` giờ lưu metadata trong comment HTML để file vẫn đọc được bằng mắt:
`- **Nơi ở**: Huế <!-- conf=0.94 n=3 seen=87 corr=1 -->`
Phần metadata chỉ nằm trên đĩa. Prompt nhận bản sạch từ `render_for_prompt()` (không metadata, không fact đã decay), nên bonus không cộng thêm token vào prompt.

### A. Confidence threshold (`confidence_threshold=0.6`)
- **Giải quyết:** extractor luôn trả về candidate kèm điểm tin cậy: khẳng định rõ như "tên mình là…" ~0.95, "ở X" mơ hồ 0.75, **câu đùa / chuyến công tác / ví dụ cũ chỉ 0.3**. Chỉ candidate đạt ngưỡng mới được ghi. Ngưỡng nằm ở một chỗ (`apply_candidates`), nên dễ chỉnh và dễ kiểm thử.
- **Cải thiện (đo được):** bảng ablation, Standard: không gate 98.6% recall (conv-10 nhắc "Đà Nẵng như ví dụ cũ" ghi đè Huế), có gate **100%**. Stress: cả hai đều 100%, nhưng không gate bị 5 lần ghi đè nhiễu so với 1 lần ở có gate (2 candidate nhiễu bị từ chối). Test `test_confidence_threshold_blocks_noise` mô phỏng đúng kịch bản "Hà Nội / product manager".
- **Rủi ro:** ngưỡng quá cao sẽ bỏ sót fact thật nói bằng giọng nhẹ nhàng (false negative). Confidence đến từ luật regex, không phải xác suất đã hiệu chuẩn. Muốn production cần dùng model phân loại hoặc hỏi xác nhận người dùng.

### B. Conflict handling
- **Giải quyết:** một key chỉ có một giá trị hiện tại. Câu phủ định ("không còn làm backend", "chứ không còn ở Đà Nẵng", "lúc đầu…") bị loại khỏi candidate, và giá trị mới thay giá trị cũ, đồng thời tăng bộ đếm `corr`.
- **Cải thiện:** recall câu hỏi "nghề hiện tại" trả đúng MLOps engineer, không lẫn backend. `test_conflict_handling_keeps_single_current_fact` kiểm tra file chỉ có một dòng `Nơi ở` và không còn chuỗi "Đà Nẵng". Không giữ bản cũ cũng giữ cho file nhỏ.
- **Rủi ro:** "bản mới nhất thắng" nghĩa là một correction **sai** (hoặc hiểu nhầm) sẽ xóa fact đúng mà không có đường quay lại. Hiện chỉ có bộ đếm `corr`, chưa có lịch sử khôi phục. Nên thêm audit log hoặc xác nhận với người dùng với fact quan trọng.

### C. Memory decay (`decay_half_life_turns=200`, `decay_min_score=0.25`)
- **Giải quyết:** fact không được nhắc lại thì điểm giảm theo `conf × 0.5^(tuổi / (half_life × (1 + ln(số lần nhắc))))`. Fact nhắc nhiều lần giảm chậm hơn, và `Tên` được miễn decay. Đồng hồ là bộ đếm lượt logic của từng user (không dùng giờ thật) nên benchmark vẫn lặp lại được.
- **Cải thiện:** fact dưới ngưỡng bị loại khỏi prompt, vừa giảm token nền vừa tránh dùng thông tin lỗi thời. `test_memory_decay_drops_stale_facts_but_keeps_identity` cho thấy sau 40 lượt không nhắc (half-life 5), prompt chỉ còn `Tên`. Fact nhắc 8 lần vẫn sống sau 12 lượt. Fact vẫn còn trên đĩa để audit. Với benchmark hiện tại (≤115 lượt/user) decay chưa kích hoạt, vì half-life 200 được chọn để không làm mất fact của bài.
- **Rủi ro:** nếu half-life quá ngắn, agent "quên" fact đúng nhưng ít nhắc (ví dụ tên thú cưng). Nếu quá dài thì decay vô nghĩa. Số 200 lượt là giả định, chưa tinh chỉnh trên dữ liệu thật. Cũng cần nhớ: "ít nhắc" không đồng nghĩa "sai".

### D. Entity extraction
- **Giải quyết:** thay vì ghi cả câu, fact được tách thành field có kiểu: `Tên`, `Nơi ở`, `Nghề nghiệp`, `Đồ uống / Món ăn yêu thích`, `Thú cưng` (`corgi tên Bơ`), `Style trả lời` và `Mối quan tâm` (danh sách có trần 5 mục). Câu hỏi, câu "nhắc lại" bị bỏ qua để không ghi nhầm.
- **Cải thiện:** người dùng đổi nghề chỉ sửa đúng một dòng, nên phần correction và decay hoạt động chính xác theo từng field. Trả lời recall ghép đúng field theo từ khóa câu hỏi nên ngắn (`Agent tokens only` thấp). Tách "AI" (chủ đề) khỏi "ai" (đại từ) bằng regex phân biệt hoa/thường (`test_entity_extraction_structured_and_pronoun_disambiguation`).
- **Rủi ro:** schema cố định làm mất fact không nằm trong schema (ví dụ "chạy bộ lúc 6 giờ" bị bỏ). Extractor là luật tiếng Việt viết tay, dễ vỡ với cách diễn đạt mới. Có thể đạt 100% trên bộ dữ liệu này mà vẫn kém trên dữ liệu thật. Production nên dùng LLM extraction kèm schema, và giữ bộ luật làm kiểm tra chéo.

---

## 4. Hạn chế còn lại (nói thẳng)
- Benchmark chạy **offline** với agent giả lập deterministic, nên chứng minh được cơ chế và chi phí ngữ cảnh, không chứng minh chất lượng của một LLM thật. Chế độ live (LangGraph) có sẵn nhưng chưa được benchmark.
- `estimate_tokens` là heuristic 4 ký tự/token, tiếng Việt thực tế thường tốn nhiều token hơn. Tỉ lệ tương đối (-71%) đáng tin hơn số tuyệt đối.
- Dữ liệu Standard có 10 hội thoại của một user nên recall dễ đạt 100%. Con số 100% không nên đọc là "hệ thống hoàn hảo".
- Summary của compact chỉ lấy câu đầu từng tin và các câu "việc còn mở". Với tin dài, thông tin sau câu đầu bị mất khi compact (đã giải thích ở 2.3).
