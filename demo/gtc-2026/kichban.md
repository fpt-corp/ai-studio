# Nvidia GTC 2026 - FPT Smartcloud Booth

---

## Demo 1: Máy Ảo GPU - Giải Pháp Phát Triển Ứng Dụng AI Linh Hoạt

**Tên Sản Phẩm:** GPU Virtual Machine

**Giá Trị Kinh Doanh:**
- Triển khai môi trường phát triển trong vài phút mà không cần cơ sở hạ tầng phức tạp
- Linh hoạt lựa chọn cấu hình phù hợp từ 1 đến 8 GPUs tùy theo nhu cầu của dự án
- Mô hình chi trả linh hoạt (PAYG - Pay As You Go) theo GPU-second, giúp tối ưu chi phí

**Kịch Bản Thực Hiện:**
1. Tạo máy ảo GPU với các cấu hình khác nhau (1, 2, 8 GPUs)
2. Hiển thị thông tin GPU và hiệu năng (nvidia-smi, nvidia-smi topo -m cho hệ thống 8 GPUs)
3. Khởi động JupyterLab và chạy mô hình AI thực tế
4. Thực hiện fine-tune và inference trên Large Language Model

**Điểm Nổi Bật:**
- Môi trường phát triển sẵn sàng sử dụng ngay lập tức
- Hỗ trợ nhiều mục đích sử dụng: coding, huấn luyện, model serving
- Giá cạnh tranh so với các nhà cung cấp khác

**Công Nghệ NVIDIA:** GPU Operator, NVLink, DCGM

---

## Demo 2: GPU Cluster - Giải Pháp Huấn Luyện AI Quy Mô Lớn

**Tên Sản Phẩm:** Kubernetes + GPU + SLURM (Managed Compute Cluster)

**Giá Trị Kinh Doanh:**
- Tối ưu hóa cho huấn luyện mô hình ngôn ngữ lớn (Large Scale LLM) trên cơ sở hạ tầng được quản lý
- Cho phép nhiều thành viên trong team chia sẻ và sử dụng GPU cluster một cách hiệu quả
- Mở rộng quy mô trong vài phút với thay đổi mã tối thiểu
- Giám sát hiệu suất thực thời cho toàn bộ cụm và các công việc

**Kịch Bản Thực Hiện:**
1. Trình bày cụm Kubernetes + SLURM đã chuẩn bị (4 nodes với môi trường đầy đủ)
2. Quản lý cụm thông qua FPT Cloud Console và Lens dashboard
3. Gửi và giám sát các công việc huấn luyện trên 2 nodes
4. Chạy nhiều công việc huấn luyện song song
5. Hiển thị danh sách công việc và tình trạng giám sát trên hệ thống Fmon

**Điểm Nổi Bật:**
- Cụm GPU được tối ưu, môi trường lý tưởng cho đội ngũ AI huấn luyện mô hình quy mô lớn
- Tối ưu hiệu quả sử dụng tài nguyên cụm thông qua Slurm orchestrator
- Giám sát tài nguyên và tiến trình công việc trong thời gian thực với FPTCloud Monitoring
- Hỗ trợ 24/7

**Công Nghệ NVIDIA:** GPU Operator, InfiniBand, DCGM

---

## Demo 3: Model as a Service - AI Marketplace

**Tên Sản Phẩm:** Model as a Service (MaaS)

**Giá Trị Kinh Doanh:**
- Tích hợp nhanh chóng các mô hình AI phù hợp vào ứng dụng mà không cần đầu tư hạ tầng, tự serving
- Giá cạnh tranh và đáng tin cậy so với các nhà cung cấp dịch vụ mô hình (ví dụ: OpenAI)
- Khả năng mở rộng tự động cho phép xử lý nhu cầu tăng đột ngột

**Kịch Bản Thực Hiện:**
1. Lựa chọn mô hình phù hợp từ FPT AI Marketplace cho các ứng dụng phổ biến (RAG, AI Agent, v.v.):
1.1. Sử dụng model Qwen3 coder cho devops agent (OpenClaw) hỗ trợ check cụm issue trong cụm k8s...
2.2. Sử dụng model GLM 4.7 cho chat with pdf RAG app
2. Lấy API key và endpoint từ hệ thống
3. Tích hợp vào ứng dụng và kiểm tra hoạt động
4. Trình bày chi tiết sử dụng token, chi phí thực tế và so sánh với các dịch vụ cạnh tranh

**Điểm Nổi Bật:**
- Tích hợp nhanh chóng, linh hoạt (Plug and Play)
- Cho phép lựa chọn mô hình phù hợp cho từng ứng dụng hoặc AI Agent
- Giá cạnh tranh và tính toán chi phí minh bạch
- Độ tin cậy cao và mở rộng quy mô tự động

**Công Nghệ NVIDIA:** Model optimization, CUDA
