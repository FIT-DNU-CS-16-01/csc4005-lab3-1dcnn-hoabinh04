# CSC4005 Lab 3 Report – UrbanSound8K with 1D-CNN

## 1. Thông tin sinh viên

- Họ tên: Nguyễn Hòa Bình
- Mã sinh viên: 1671040004
- Lớp: KHMT 16-01
- Link GitHub repo: https://github.com/hoabinh04/csc4005-lab3-urbansound-1dcnn
- Link W&B run/project: https://wandb.ai/1671040004-dai-nam/csc4005-lab3-urbansound-1dcnn

---

## 2. Mục tiêu thí nghiệm

Mô tả ngắn gọn mục tiêu của lab:

- Thực hiện phân loại 10 loại âm thanh môi trường từ tập dữ liệu UrbanSound8K sử dụng kiến trúc mạng nơ-ron tích chập 1 chiều (1D-CNN).
- Sử dụng các đặc trưng âm học như MFCC và Log-mel Spectrogram để làm dữ liệu đầu vào, thay vì sử dụng tín hiệu thời gian thô (Raw Waveform).
- Xây dựng pipeline tiền xử lý và huấn luyện mô hình tự động, theo dõi các chỉ số quan trọng (loss, accuracy) thông qua Weights & Biases (W&B).
- Phân tích hiệu năng mô hình qua Learning Curves và Confusion Matrix để hiểu rõ các lỗi nhầm lẫn của mô hình.

---

## 3. Dữ liệu và tiền xử lý

### 3.1. Dataset

- Dataset: UrbanSound8K
- Số lớp: 10
- Các lớp: ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
- Fold dùng để train: 1, 2, 3, 4, 5, 6, 7, 8
- Fold dùng để validation: 9
- Fold dùng để test: 10

### 3.2. Tiền xử lý audio

Điền cấu hình đã dùng:

| Thành phần | Giá trị |
|---|---|
| Sample rate | 16000 |
| Duration | 4.0s |
| Feature type | mfcc / logmel |
| n_mfcc / n_mels | 40 / 64 |
| n_fft | 1024 |
| hop_length | 512 |
| Augmentation | true (TimeShift, GaussianNoise) |

Giải thích ngắn: Việc đưa audio về cùng sample rate giúp đảm bảo các đặc trưng tần số được trích xuất nhất quán trên mọi mẫu. Việc đưa về cùng độ dài 4 giây (pad hoặc crop) giúp tạo ra các batch dữ liệu có kích thước đồng nhất, giúp mô hình tích chập 1D xử lý được các chuỗi đầu vào có cùng số lượng frames.

---

## 4. Mô hình 1D-CNN

Mô tả kiến trúc mô hình:

```text
Input feature sequence (Batch, Features, Frames)
→ Conv1D Block 1 (Channels: 40/64 -> 64, Kernel: 3, ReLU, BatchNorm, Dropout)
→ Conv1D Block 2 (Channels: 64 -> 128, Kernel: 3, ReLU, BatchNorm, Dropout)
→ Conv1D Block 3 (Channels: 128 -> 128, Kernel: 3, ReLU, BatchNorm, Dropout)
→ Global Average Pooling (Dọc theo trục thời gian)
→ Dense classifier (128 -> 10)
→ Softmax
```

Bảng cấu hình:

| Thành phần | Giá trị |
|---|---|
| model_name | mfcc_1dcnn / logmel_1dcnn |
| hidden_channels | 64 |
| dropout | 0.3 |
| optimizer | adamw |
| learning rate | 0.001 |
| weight decay | 1e-4 |
| batch size | 32 |
| epochs | 12 |
| patience | 5 |

---

## 5. Kết quả thực nghiệm

### 5.1. Kết quả chính

| Metric | Giá trị (Baseline MFCC) | Giá trị (Log-mel) |
|---|---:|---:|
| Best validation accuracy | 60.69% | 65.66% |
| Test accuracy | 53.55% | 64.09% |
| Average epoch time | 6.12s | 4.28s |
| Total parameters | 137,930 | 145,610 |
| Trainable parameters | 137,930 | 145,610 |

### 5.2. Learning curves

Chèn hình `curves.png`.

Nhận xét:
- Train loss và val loss đều giảm ổn định trong các epoch đầu.
- Có dấu hiệu overfitting nhẹ ở baseline MFCC khi train accuracy đạt gần 98% nhưng val accuracy chững lại ở mức 60%. Biến thể Log-mel cho kết quả cân bằng hơn.
- Early stopping đã xảy ra ở epoch 11 đối với baseline MFCC để tránh mô hình quá khớp.

### 5.3. Confusion matrix

Chèn hình `confusion_matrix.png`.

Nhận xét:
- Những lớp dễ phân loại: siren, gun_shot do có pattern âm thanh đặc trưng, tách biệt.
- Những lớp dễ bị nhầm: air_conditioner nhầm với engine_idling, street_music nhầm với children_playing.
- Nguyên nhân: Các âm thanh như máy điều hòa và động cơ nổ không tải có dải tần năng lượng khá giống nhau (tiếng ù thấp). Street music và tiếng trẻ em chơi đùa thường chứa nhiều tạp âm nền đô thị tương tự nhau.

---

## 6. W&B tracking

Dán link W&B:

```text
https://wandb.ai/1671040004-dai-nam/csc4005-lab3-urbansound-1dcnn
```

Ảnh chụp dashboard bao gồm các biểu đồ learning curves, metrics cuối cùng và confusion matrix được log tự động cho từng run.

---

## 7. Phân tích và thảo luận

1. Vì sao dùng 1D-CNN thay vì MLP cho chuỗi đặc trưng audio?
   - 1D-CNN có khả năng học các pattern cục bộ theo thời gian và chia sẻ tham số, giúp nhận diện các đặc trưng âm thanh ngắn hạn hiệu quả và mạnh mẽ hơn MLP.
2. Kernel 1D trong bài này đang trượt theo chiều nào?
   - Kernel trượt dọc theo chiều thời gian (trục frames).
3. MFCC giúp mô hình học dễ hơn raw waveform ở điểm nào?
   - MFCC nén dữ liệu từ hàng vạn điểm tín hiệu thô về vài chục kênh đặc trưng phổ, loại bỏ nhiễu và tập trung vào các thành phần năng lượng quan trọng mang tính chất âm học.
4. Mô hình hiện tại còn hạn chế gì?
   - Độ chính xác trên tập test chưa thực sự cao (~64%), vẫn còn hiện tượng overfitting do kích thước tập dữ liệu huấn luyện còn hạn chế và sự tương đồng giữa một số lớp âm thanh.
5. Có thể cải thiện kết quả bằng cách nào?
   - Tăng cường dữ liệu (SpecAugment), sử dụng các kiến trúc sâu hơn, hoặc thử nghiệm các thuật toán tối ưu khác.

---

## 8. Bài mở rộng nếu có

| Pipeline | Feature/Input | Test accuracy | Nhận xét |
|---|---|---:|---|
| Baseline | MFCC + 1D-CNN | 53.55% | Hiệu năng ổn định, thời gian trích xuất đặc trưng nhanh. |
| Extension 1 | log-mel + 1D-CNN | 64.09% | Kết quả vượt trội, giữ được cấu trúc phổ chi tiết hơn. |

---

## 9. Kết luận

- Đã xây dựng thành công pipeline phân loại âm thanh hoàn chỉnh từ tiền xử lý đến đánh giá.
- Thấy rõ ưu điểm của việc sử dụng các đặc trưng phổ (MFCC, Log-mel) so với tín hiệu thô.
- Nắm vững cách sử dụng W&B để quản lý nhiều phiên bản thí nghiệm khác nhau.
- Rút ra kết luận Log-mel Spectrogram mang lại độ chính xác cao hơn MFCC cho bài toán UrbanSound8K với kiến trúc 1D-CNN này.
