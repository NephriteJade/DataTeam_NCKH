# QUY TRÌNH XỬ LÝ DỮ LIỆU VLOG DATASET
Quy trình xử lý dữ liệu vlog được thực hiện theo 8 bước tuần tự, mỗi bước đóng vai trò quan trọng trong việc đảm bảo chất lượng dữ liệu đầu ra. Việc xử lý toàn diện này là nền tảng cốt lõi cho các bài toán AI như nhận diện cảm xúc, phân tích nội dung, xử lý ngôn ngữ tự nhiên và nhận diện hành vi.
# Bước 1: Thu Thập Dữ Liệu
-	Nguồn cung cấp: Dữ liệu có thể được thu thập từ các nền tảng trực tuyến, các dataset đã được công bố từ cộng đồng nghiên cứu hoặc trực tiếp tạo các dữ liệu bằng việc đi quay cái video.
-	Tiêu chuẩn đầu vào: Cần đảm bảo video có chất lượng cao về độ phân giải và âm thanh, thống nhất định dạng thu thập, đồng thời tuân thủ chặt chẽ các chính sách bản quyền dữ liệu (tránh vi phạm bản quyền hoặc sử dụng các dữ liệu cần phải được cho phép)

# Bước 2: Tiền Xử Lý Video
-	Chuẩn hóa: Chuyển đổi và thống nhất các video về định dạng MP4 H.264.
-	Trích xuất đặc trưng: Tiến hành cắt video thành các đoạn nhỏ hơn (segmentation) và trích xuất các khung hình (frame extraction) theo một tần suất nhất định.
-	Mục đích: Giảm tải kích thước lưu trữ và chuẩn bị dữ liệu đầu vào trực tiếp cho các mô hình thị giác máy tính.

# Bước 3: Xử Lý Âm Thanh
-	Làm sạch: Tách âm thanh ra khỏi video và áp dụng các kỹ thuật loại bỏ nhiễu (noise reduction).
-	Chuẩn hóa: Đưa tần số lấy mẫu về tiêu chuẩn chung, thường là 16kHz hoặc 44.1kHz.
-	Phân vùng: Áp dụng kỹ thuật tách đoạn lời nói (Voice Activity Detection - VAD) để khoanh vùng khu vực có tiếng người, hỗ trợ quá trình phân tích giọng nói.

# Bước 4: Xử Lý Văn Bản (Transcript)
-	Tạo phụ đề: Transcript được sinh tự động thông qua các công cụ speech-to-text (như Whisper, Google STT), AI để bóc tách hoặc trực tiếp nhập thủ công.
-	Làm sạch và chuẩn hóa: Loại bỏ các thẻ HTML, ký tự đặc biệt; thống nhất việc sử dụng chữ hoa/thường và dấu câu.
-	Tiền xử lý NLP: Thực hiện tách từ (tokenization) và các kỹ thuật xử lý ngôn ngữ tự nhiên khác.
·  Làm sạch văn bản (Text Cleaning): Loại bỏ các dữ liệu nhiễu không cần thiết cho quá trình phân tích như các thẻ HTML, URL liên kết, biểu tượng cảm xúc (emoji), dấu câu, hoặc các ký tự đặc biệt.
         ·  Chuẩn hóa chữ viết (Lowercasing): Đưa toàn bộ văn bản về chữ in thường. Bước này giúp đồng nhất dữ liệu, để máy tính hiểu rằng "Video" và "video" chỉ là một từ duy nhất, tránh việc phải học hai lần cho cùng một khái niệm.
	·  Tách từ (Tokenization): Cắt một đoạn văn bản dài thành các đơn vị nhỏ hơn (gọi là token), thường là từng từ hoặc cụm từ.  
Ví dụ: Câu "Dữ liệu rất quan trọng" sẽ được tách thành danh sách các token: ["Dữ", "liệu", "rất", "quan", "trọng"] hoặc ["Dữ_liệu", "rất", "quan_trọng"].
·  Loại bỏ từ dừng (Stopword Removal): Lọc bỏ những từ ngữ xuất hiện với tần suất quá dày đặc nhưng lại mang rất ít giá trị về mặt ngữ nghĩa cốt lõi của câu (ví dụ trong tiếng Việt là các từ: "là", "thì", "mà", "bởi", "những"...). Việc này giúp giảm kích thước bộ nhớ và giữ lại những từ khóa mang thông tin thực sự.
	·  Chuẩn hóa từ gốc (Stemming / Lemmatization): Đưa các biến thể ngữ pháp của một từ về dạng nguyên bản. Kỹ thuật này đặc biệt quan trọng với các ngôn ngữ có chia động từ như tiếng Anh (ví dụ: chuyển đổi "running", "ran", "runs" về chung một gốc là "run").

# Bước 5: Gán Nhãn Dữ Liệu (Annotation)
* Tổng Quan:
Gán nhãn là bước con người "dạy" cho máy biết mỗi đoạn dữ liệu là gì. Đây là bước tốn công sức nhất nhưng cũng quan trọng nhất — nhãn sai thì model học sai, dù các bước trước có làm tốt đến đâu.
# 5.1. Định Nghĩa Nhãn
Trước khi gán nhãn, phải xác định rõ model cần học cái gì. Có 3 loại nhãn phổ biến trong dataset vlog:
-	Nhãn hành động — nhận diện người đang làm gì trong video:
Mã nhãn	Tên nhãn	Mô tả
0	(nói chuyện)	Người đang phát âm liên tục, nhìn vào camera
1	(đi lại)	Người đang di chuyển trong khung hình
2	(giới thiệu sản phẩm)	Người cầm/chỉ vào vật thể và mô tả
-	Nhãn chủ đề — phân loại nội dung vlog thuộc lĩnh vực gì: du lịch, ẩm thực, công nghệ, làm đẹp...
-	Nhãn cảm xúc — nhận diện trạng thái cảm xúc: vui, buồn, tức giận, trung tính, phấn khích...
# 5.2. Ví Dụ Thực Tế
- Giả sử có 1 video vlog dài 10 phút, được cắt thành các đoạn 5 giây. Người gán nhãn xem từng đoạn và điền vào bảng:
Đoạn	Nội dung	Hành động	Cảm xúc	Chủ đề
0:00–0:05	"Xin chào mọi người!"	nói chuyện	vui vẻ	làm đẹp
0:05–0:10	Đi vào phòng quay	đi lại	trung tính	làm đẹp
0:10–0:15	Cầm son môi lên giới thiệu	giới thiệu SP	phấn khích	làm đẹp
0:15–0:20	Ngồi im, nhìn ra ngoài	không xác định	buồn	làm đẹp

- Sau khi gán nhãn xong, dữ liệu được lưu dạng JSON như sau:
{
  "video_id": "vlog_001",
  "segment": "0:10-0:15",
  "action": "gioi thieu san pham",
  "emotion": "phan khich",
  "topic": "lam dep"
}
# 5.3. Kiểm Soát Chất Lượng
Gán nhãn do con người làm nên dễ sai và không nhất quán. Cần 3 biện pháp kiểm soát:
# a) Viết tài liệu hướng dẫn rõ ràng (Annotation Guideline)
Định nghĩa chính xác từng nhãn để tránh hiểu nhầm. Ví dụ: "nói chuyện" chỉ được gán khi người phát âm liên tục ít nhất 2 giây, không phải chỉ mở miệng.
# b) Dùng nhiều người gán nhãn (Inter-annotator Agreement)
Cùng một đoạn video giao cho 2–3 người gán nhãn độc lập, sau đó so sánh kết quả:
•	Nếu đồng thuận cao → nhãn đáng tin cậy 
•	Nếu đồng thuận thấp → định nghĩa nhãn đang mơ hồ, cần làm rõ lại 
Chỉ số đo mức đồng thuận thường dùng là Cohen's Kappa hoặc Fleiss' Kappa.
# c) Kiểm tra định kỳ
Sau khi gán nhãn hàng loạt, lấy mẫu ngẫu nhiên ~5–10% để kiểm tra lại, tránh tình trạng người gán nhãn làm ẩu về sau.

# Bước 6: Đồng Bộ Dữ Liệu
-	Khớp nối: Đây là công đoạn mang tính quyết định, đảm bảo mọi thành phần bao gồm video, âm thanh (audio), văn bản (transcript) và nhãn (label) phải khớp chính xác với nhau về mặt thời gian (timestamp).
-	Mục đích: Ngăn ngừa tình trạng sai lệch đồng bộ, nguyên nhân chính dẫn đến sự suy giảm nghiêm trọng về hiệu suất của mô hình.
-	
# Bước 7: Chuẩn Hóa Dữ Liệu
Toàn bộ tập dữ liệu cần được đưa về một quy chuẩn chung:
•	Video: Độ phân giải 720p hoặc 1080p, tốc độ 25fps, lưu dưới định dạng MP4.
•	Audio: Tần số 16kHz, kênh Mono/Stereo, định dạng WAV.
•	Text: Bảng mã UTF-8, lưu trữ dạng file TXT hoặc JSON.
•	Label: Sử dụng mã hóa đồng nhất dạng integer encoding hoặc One-hot.

# Bước 8: Lưu Trữ Dữ Liệu
# 8.1. Tổ Chức Thư Mục
- Thay vì để file lung tung, bạn cần xây dựng cấu trúc thư mục rõ ràng, ví dụ:
vlog_dataset/
├── raw/          ← video gốc chưa xử lý
├── video/        ← video đã chuẩn hóa (MP4, 720p/1080p)
├── audio/        ← file âm thanh đã tách (WAV, 16kHz)
├── transcript/   ← file văn bản/phụ đề (TXT, JSON)
└── labels/       ← nhãn dữ liệu (integer/one-hot)
-	Mục tiêu: Ai cũng có thể tìm đúng file nhanh chóng, không bị nhầm lẫn giữa các phiên bản.
# 8.2. Quản Lý Tập Trung bằng File Index/Manifest
=-	Đây là một file tổng hợp (thường dạng CSV hoặc JSON) liệt kê toàn bộ các mẫu dữ liệu trong dataset, ví dụ:
[
  {
    "id": "video_001",
    "video": "video/video_001.mp4",
    "audio": "audio/video_001.wav",
    "transcript": "transcript/video_001.json",
    "label": 2,
    "duration": 35.4
  },
  ...
]
- Khi huấn luyện model, code chỉ cần đọc file manifest này thay vì phải dò từng thư mục — vừa nhanh, vừa tránh sót file.
# 8.3. Bảo Trì: Version Control & Sao Lưu
•	Version control: Giống như Git cho code, bạn cần đánh phiên bản cho dataset (v1.0, v1.1, v2.0...) để biết dữ liệu nào đã được thêm, sửa, hay xóa theo thời gian. Công cụ phổ biến là DVC (Data Version Control).
•	Sao lưu định kỳ: Dataset thường rất lớn và tốn nhiều công sức tạo ra — nếu mất là mất trắng. Cần backup lên cloud (S3, Google Drive...) hoặc ổ cứng dự phòng.
