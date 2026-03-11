# Nvidia GTC 2026 - FPT Smartcloud Booth

---
## Demo 1: AIStudio - Quay video trước / demo trực tiếp
**Tên Sản Phẩm:** AIStudio

**Kịch Bản Thực Hiện:**
Sử dụng kịch bản tutorial health care chatbot - BYOAI

**Điểm Nổi Bật:**
- Node-code environment cho finetune/evaluate/manage AI model
- PAYG, chỉ tính tiền trên thời gian training


## Demo 2: Máy Ảo GPU - Giải Pháp Phát Triển Ứng Dụng AI Linh Hoạt

**Tên Sản Phẩm:** GPU Virtual Machine

**Giá Trị Kinh Doanh:**
- Triển khai môi trường phát triển trong vài phút mà không cần cơ sở hạ tầng phức tạp
- Linh hoạt lựa chọn cấu hình phù hợp từ 1 đến 8 GPUs tùy theo nhu cầu của dự án
- Mô hình chi trả linh hoạt (PAYG - Pay As You Go) theo GPU-second, giúp tối ưu chi phí

**Kịch Bản Thực Hiện:**
1. Tạo máy ảo GPU với các cấu hình khác nhau (1, 2, 8 GPUs)
2. Hiển thị thông tin GPU và hiệu năng (nvidia-smi, nvidia-smi topo -m cho hệ thống 8 GPUs)
3. Khởi động JupyterLab và cài pytorch
4. Check pytorch cuda available

**Điểm Nổi Bật:**
- Môi trường phát triển sẵn sàng sử dụng ngay lập tức
- Hỗ trợ nhiều mục đích sử dụng: coding, huấn luyện, model serving
- Giá cạnh tranh so với các nhà cung cấp khác

**Công Nghệ NVIDIA:** GPU Operator, NVLink, DCGM

---

## Demo 3: GPU Cluster - Giải Pháp Huấn Luyện AI Quy Mô Lớn -> Chuẩn bị sẵn và quay video

**Tên Sản Phẩm:** Kubernetes + GPU + SLURM (Managed Compute Cluster)

**Giá Trị Kinh Doanh:**
- Tối ưu hóa cho huấn luyện mô hình ngôn ngữ lớn (Large Scale LLM) trên cơ sở hạ tầng được quản lý
- Cho phép nhiều thành viên trong team chia sẻ và sử dụng GPU cluster một cách hiệu quả
- Mở rộng quy mô trong vài phút với thay đổi mã tối thiểu
- Giám sát hiệu suất thực thời cho toàn bộ cụm và các công việc

**Kịch Bản Thực Hiện:**
1. Trình bày cụm Kubernetes + SLURM đã chuẩn bị (4 nodes với môi trường đầy đủ)
2. Quản lý cụm thông qua FPT Cloud Console và Lens dashboard
3. Truy cập vào Slurm login node qua ssh: `ssh root@161.248.3.119`
3. Chạy thử các lệnh Slurm để kiểm tra thông tin hệ thống:
   - `sinfo` (xem partition và trạng thái node)
   - `sinfo -N -l` (xem chi tiết từng node)
   - `scontrol show nodes` (xem cấu hình và tài nguyên node)
   - `squeue` (xem hàng đợi job hiện tại)
   - `squeue -u $USER` (lọc job của user đang demo)
   - `sacct -S today` (xem lịch sử job trong ngày)
   - `srun --pty -N1 -n1 bash` (chạy interactive session để kiểm tra nhanh môi trường)
   - `srun --pty -N1 -n1 nvidia-smi` (chạy session không có GPU)
  - `srun --pty -N1 -n1 --gres=gpu:1 nvidia-smi` (chạy session có 1 GPU)
4. Chạy job training PyTorch đơn giản trên 2 nodes:
   - Clone source code: `git clone  https://github.com/fpt-corp/ai-studio.git`
   - Di chuyển tới thư mục GTC demo 3: `cd /root/ai-studio/demo/gtc-2026/demo3` 
   - Script training DDP mẫu: [demo3/ddp_demo.py](demo3/ddp_demo.py)
   - Script setup môi trường bằng uv: [demo3/setup_env_uv.sh](demo3/setup_env_uv.sh)
   - Script submit Slurm: [demo3/pt-2node.sbatch](demo3/pt-2node.sbatch)
   - Setup môi trường chạy:
     - `bash setup_env_uv.sh`
   - Submit job và tự lấy `job_id`:
     - `JOB_TRAIN=$(sbatch --parsable pt-2node.sbatch)`
   - Theo dõi nhanh:
     - `echo "JOB_TRAIN=$JOB_TRAIN"`
     - `squeue -j $JOB_TRAIN` (xác nhận `PD`/`R`)
     - `scontrol show job $JOB_TRAIN` (kiểm tra node phân bổ)
     - `tail -f slurm-$JOB_TRAIN.out` (xem output train từ 2 nodes)
5. Hiển thị danh sách công việc và tình trạng giám sát trên hệ thống Fmon

**Điểm Nổi Bật:**
- Cụm GPU được tối ưu, môi trường lý tưởng cho đội ngũ AI huấn luyện mô hình quy mô lớn
- Tối ưu hiệu quả sử dụng tài nguyên cụm thông qua Slurm orchestrator
- Giám sát tài nguyên và tiến trình công việc trong thời gian thực với FPTCloud Monitoring
- Hỗ trợ 24/7

**Công Nghệ NVIDIA:** GPU Operator, InfiniBand, DCGM

## Demo 4: Model as a service, FPT Flagship model for document QA and parsing

**Tên Sản Phẩm:** Enterprise LLM model, Model as a Service (MaaS)

**Giá Trị Kinh Doanh:**
- Flagship Model do FPT phát triển TỪ ĐẦU (pretrain -> finetune), độ chính xác vượt trội trên các usecase đặc thù: OCR, layout analysis, table parsing
- Tích hợp nhanh chóng các mô hình AI phù hợp vào ứng dụng mà không cần đầu tư hạ tầng, tự serving
- Giá cạnh tranh và đáng tin cậy so với các nhà cung cấp dịch vụ mô hình (ví dụ: OpenAI)

**Kịch Bản Thực Hiện:**
1. Show giao diện marketplace
    - Trang chủ
    - Danh sách model với đa dạng model Opensource
    - Có cả các model do FPT tự phát triển với độ chính xác vượt trội: show model latest
2. Show kết quả đánh giá mô hình do FPT tự phát triển (lấy từ slide .pptx)
3. App demo: UI show input và output
    - Input: là file PDF ảnh
    - Output: Text chunk đã được trích xuất
    - Highlights: Trích xuất dữ liệu trên bảng, tự động mô tả biểu đồ

**Điểm Nổi Bật:**
- Flagship model độ chính xác vượt trội
- Tích hợp nhanh chóng, linh hoạt (Plug and Play)
- Cho phép lựa chọn mô hình phù hợp cho từng ứng dụng hoặc AI Agent
- Giá cạnh tranh và tính toán chi phí minh bạch
- Độ tin cậy cao và mở rộng quy mô tự động

**Công Nghệ NVIDIA:** Model optimization, CUDA


## Demo 5: Enterprise tenant
**Tên Sản Phẩm:** FPT AI Factory - FPT Cloud console

**Giá Trị Kinh Doanh:**
- Quản lý resource AI dễ dàng, chuyên nghiệp
- Dễ dàng mở rộng

**Kịch Bản Thực Hiện:**
1. Show 1 tenant trên console.fptcloud.com với nhiều resource: VPC, BM, VM, subnet, user..., show dashboard quản lý tổng quan tenant/vpc
2. Show Dashboard monitor resource tổng quan, dashboard chi tiết của server (optional)
3. Show Dashboard monitor workload k8s: nodes, pods, network...

**Điểm Nổi Bật:**
- Cung cấp nhiều loại resource, CPU, GPU, high-end GPU, đáp ứng đa dạng nhu cầu, nhiều quy mô, từ nhỏ đến lớn, dễ dàng mở rộng
- Dễ dàng tạo và quản lý resource
- Builtin monitoring, logging

**Công Nghệ NVIDIA:** DCGM, HGX cluster
