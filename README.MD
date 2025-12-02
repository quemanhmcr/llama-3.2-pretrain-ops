# llama-3.2-pretrain-ops

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Kết Quả Huấn Luyện

![Kết quả huấn luyện mô hình Llama 3.2](./ket_que_train.png)

## 🚀 Tổng Quan

Dự án này cung cấp một triển khai chuyên nghiệp, mô-đun hóa và hiệu năng cao cho việc huấn luyện trước (pretraining) các mô hình ngôn ngữ theo phong cách **Llama 3.2 1B**. Được thiết kế theo các tiêu chuẩn kỹ thuật của "Big Tech", dự án bao gồm quy trình xử lý dữ liệu mạnh mẽ, kiến trúc mô hình hiệu quả và vòng lặp huấn luyện sẵn sàng cho môi trường production.

Hệ thống bao gồm các khả năng lọc dữ liệu nâng cao như **MinHash deduplication** (loại bỏ trùng lặp) và **Aho-Corasick sensitive content filtering** (lọc nội dung nhạy cảm), đảm bảo dữ liệu huấn luyện chất lượng cao từ các nguồn như OpenWebText.

## ✨ Tính Năng

- **Kiến Trúc Mô-đun**: Phân tách rõ ràng các thành phần (Configs, Data, Models, Training).
- **Kiến Trúc Llama 3.2**:
  - Rotary Positional Embeddings (RoPE)
  - SwiGLU Activation
  - RMSNorm (Root Mean Square Normalization)
  - Grouped Query Attention (GQA)
- **Quy Trình Dữ Liệu Nâng Cao**:
  - **Deduplication**: Sử dụng MinHash LSH để phát hiện văn bản gần giống nhau.
  - **Filtering**: Thuật toán Aho-Corasick để chặn từ khóa nhanh chóng.
  - **Streaming**: Sử dụng IterableDatasets để xử lý kho ngữ liệu quy mô lớn mà không gặp vấn đề về bộ nhớ.
- **Huấn Luyện Production**:
  - Mixed Precision Training (AMP)
  - Gradient Accumulation
  - Tích hợp WandB & Tensorboard
  - Quản lý Checkpoint & Early Stopping

## 📂 Cấu Trúc Dự Án

```
llama_pretrain/
├── configs/                # Quản lý cấu hình
│   └── training_config.py  # Các siêu tham số huấn luyện
├── data/                   # Quy trình xử lý dữ liệu
│   ├── dataset.py          # Triển khai PyTorch Dataset
│   ├── deduplication.py    # Logic MinHash LSH
│   ├── filtering.py        # Logic lọc nội dung
│   └── processor.py        # Xử lý văn bản thô
├── models/                 # Định nghĩa kiến trúc mô hình
│   ├── args.py             # Dataclasses cấu hình mô hình
│   ├── layers.py           # Các khối Transformer (Attention, MLP)
│   ├── modeling_llama.py   # Class mô hình chính
│   └── utils.py            # Tiện ích toán học (RoPE, v.v.)
├── training/               # Điều phối huấn luyện
│   └── trainer.py          # Vòng lặp huấn luyện và quản lý trạng thái
├── utils/                  # Các tiện ích chung
├── main.py                 # Điểm khởi chạy chính (Entry point)
├── requirements.txt        # Các thư viện phụ thuộc
└── README.md               # Tài liệu hướng dẫn
```

## 🛠️ Cài Đặt

1.  **Clone repository:**
    ```bash
    git clone https://github.com/yourusername/llama_pretrain.git
    cd llama_pretrain
    ```

2.  **Tạo môi trường ảo (khuyên dùng):**
    ```bash
    python -m venv venv
    # Trên Windows:
    venv\Scripts\activate
    # Trên Linux/Mac:
    source venv/bin/activate
    ```

3.  **Cài đặt các thư viện phụ thuộc:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Hướng Dẫn Sử Dụng

### Chạy Huấn Luyện (Training)

Bạn có thể bắt đầu quá trình huấn luyện bằng cách chạy file `main.py`. Script hỗ trợ các tham số dòng lệnh để bạn dễ dàng tùy chỉnh quá trình chạy mà không cần sửa code.

**Cú pháp cơ bản:**

```bash
python main.py --data_path <đường_dẫn_đến_file_dữ_liệu> [các_tham_số_khác]
```

**Các tham số quan trọng:**

| Tham số | Kiểu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--data_path` | `str` | **Bắt buộc** | Đường dẫn tuyệt đối hoặc tương đối đến file dữ liệu huấn luyện (định dạng .txt). |
| `--output_dir` | `str` | `./checkpoints` | Thư mục để lưu các checkpoint của mô hình. |
| `--batch_size` | `int` | `4` | Kích thước batch trên mỗi thiết bị (GPU/CPU). |
| `--epochs` | `int` | `3` | Tổng số epoch huấn luyện. |
| `--lr` | `float` | `3e-4` | Tốc độ học (Learning rate). |
| `--max_seq_len` | `int` | `1024` | Độ dài chuỗi token tối đa đầu vào. |
| `--wandb` | `flag` | `False` | Thêm cờ này nếu muốn bật logging lên Weights & Biases. |

### Ví Dụ Cụ Thể

**1. Chạy thử nghiệm nhanh với dữ liệu mẫu:**

```bash
python main.py --data_path ./data/sample_data.txt --epochs 1 --batch_size 2
```

**2. Chạy huấn luyện đầy đủ với cấu hình tùy chỉnh:**

```bash
python main.py \
    --data_path "E:/datasets/openwebtext_train.txt" \
    --output_dir "./models/llama_v1" \
    --batch_size 8 \
    --epochs 5 \
    --lr 1e-4 \
    --max_seq_len 2048 \
    --wandb
```

**3. Sử dụng Makefile (Tiện lợi):**

Nếu bạn đã cấu hình sẵn, bạn có thể dùng lệnh make (lưu ý cần sửa Makefile nếu muốn truyền tham số động):

```bash
make train
```

## 📊 Giám Sát (Monitoring)

Tiến trình huấn luyện có thể được theo dõi qua:
- **Console Output**: Hiển thị loss, learning rate và tốc độ xử lý theo thời gian thực.
- **WandB**: Nếu bật cờ `--wandb`, các biểu đồ chi tiết sẽ được đẩy lên dashboard của Weights & Biases.
- **Tensorboard**: Mặc định script cũng hỗ trợ Tensorboard, log được lưu trong thư mục `output_dir/tensorboard`.

## 🤝 Đóng Góp

Mọi đóng góp đều được hoan nghênh! Vui lòng gửi Pull Request.

1.  Fork dự án
2.  Tạo feature branch (`git checkout -b feature/TinhNangMoi`)
3.  Commit thay đổi (`git commit -m 'Thêm tính năng mới'`)
4.  Push lên branch (`git push origin feature/TinhNangMoi`)
5.  Mở Pull Request

## 📄 Giấy Phép

Dự án này được cấp phép theo giấy phép MIT - xem file [LICENSE](LICENSE) để biết thêm chi tiết.
