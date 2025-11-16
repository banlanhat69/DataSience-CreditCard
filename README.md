# **Dự Đoán Rời Bỏ Dịch Vụ Của Khách Hàng Thẻ Tín Dụng**

Dự án này tập trung vào việc phân tích dữ liệu và xây dựng mô hình học máy để dự đoán khả năng một khách hàng sẽ rời bỏ dịch vụ thẻ tín dụng. Toàn bộ quy trình từ xử lý dữ liệu đến xây dựng mô hình được thực hiện chủ yếu bằng thư viện NumPy, nhằm mục đích rèn luyện và thể hiện sự am hiểu sâu sắc về các hoạt động tính toán cốt lõi trong Khoa học Dữ liệu.

### **Giới thiệu**
*   **Mô tả bài toán:** Trong ngành tài chính-ngân hàng, việc giữ chân khách hàng cũ (customer retention) có chi phí hiệu quả hơn nhiều so với việc thu hút khách hàng mới. Do đó, việc xác định sớm những khách hàng có nguy cơ rời bỏ (churn) để đưa ra các biện pháp can thiệp kịp thời là một bài toán cực kỳ quan trọng.
*   **Động lực và ứng dụng:** Dự án này mô phỏng một bài toán thực tế, nơi một ngân hàng muốn giảm tỷ lệ khách hàng rời bỏ dịch vụ thẻ tín dụng. Mô hình dự đoán có thể được tích hợp vào hệ thống CRM (Quản lý quan hệ khách hàng) để tự động gắn cờ các khách hàng rủi ro, giúp đội ngũ chăm sóc khách hàng có thể tiếp cận và đưa ra các ưu đãi phù hợp.
*   **Mục tiêu cụ thể:**
    1.  Thực hiện phân tích dữ liệu khám phá (EDA) để tìm ra các yếu tố chính ảnh hưởng đến việc khách hàng rời bỏ.
    2.  Xây dựng một quy trình tiền xử lý dữ liệu hoàn chỉnh chỉ bằng NumPy.
    3.  Huấn luyện và đánh giá mô hình Logistic Regression để dự đoán khả năng rời bỏ.

### **Dataset**
*   **Nguồn dữ liệu:** Dữ liệu được lấy từ cuộc thi "Credit Card Customers" trên nền tảng Kaggle.
    *   **Link:** [https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
*   **Kích thước và đặc điểm:**
    *   Dữ liệu gốc bao gồm 10127 khách hàng và 23 thuộc tính.
    *   Sau quá trình lựa chọn đặc trưng và xử lý outliers, bộ dữ liệu cuối cùng được sử dụng để huấn luyện có kích thước nhỏ hơn, tập trung vào các đặc trưng có giá trị nhất.
*   **Mô tả các features (đã được chọn lọc):**
    *   `Attrition_Flag`: Biến mục tiêu (Đã rời bỏ / Khách hàng hiện tại).
    *   `Total_Trans_Ct`: Tổng số lượng giao dịch.
    *   `Total_Trans_Amt`: Tổng giá trị giao dịch.
    *   `Total_Revolving_Bal`: Số dư quay vòng trên thẻ.
    *   `Total_Ct_Chng_Q4_Q1`: Sự thay đổi về số lượng giao dịch từ Quý 4 đến Quý 1.
    *   `Total_Relationship_Count`: Số lượng sản phẩm của ngân hàng mà khách hàng đang sử dụng.
    *   `Months_Inactive_12_mon`: Số tháng không hoạt động trong 12 tháng gần nhất.
    *   `Credit_Limit`: Hạn mức tín dụng của thẻ.
    *   `Avg_Utilization_Ratio`: Tỷ lệ sử dụng thẻ trung bình.

### **Phương pháp thực hiện**

Quy trình xử lý và mô hình hóa dữ liệu được chia thành các bước chính sau:

1.  **Phân tích dữ liệu khám phá (EDA):** Sử dụng Matplotlib và Seaborn để trực quan hóa dữ liệu, tìm ra các mối quan hệ và các yếu tố dự báo mạnh nhất.
2.  **Lựa chọn đặc trưng:** Dựa trên EDA, loại bỏ các đặc trưng nhân khẩu học (tuổi, giới tính) có ít ảnh hưởng và giữ lại các đặc trưng về hành vi giao dịch.
3.  **Tiền xử lý dữ liệu:**
    *   **Encoding:** Chuyển đổi các biến phân loại sang dạng số.
    *   **Kỹ thuật đặc trưng:** Tạo ra đặc trưng mới `Utilization_Ratio` (`Total_Revolving_Bal / Credit_Limit`) để tăng cường sức mạnh dự báo.
    *   **Xử lý Outliers:** Sử dụng phương pháp IQR để xác định và loại bỏ các hàng chứa giá trị ngoại lệ.
    *   **Chuẩn hóa:** Dùng Z-score scaling để đưa tất cả các đặc trưng về cùng một thang đo (trung bình 0, độ lệch chuẩn 1).
4.  **Mô hình hóa:**
    *   **Thuật toán:** Sử dụng **Logistic Regression**, một mô hình tuyến tính hiệu quả cho bài toán phân loại nhị phân. Hàm Sigmoid được dùng để chuyển đổi đầu ra thành xác suất:
        $P(y=1|X) = \sigma(z) = \frac{1}{1 + e^{-z}}$
        Trong đó $z = wX + b$.
    *   **Cài đặt bằng NumPy:** Lớp `LogisticRegression` trong `src/models.py` đã được xây dựng, cài đặt thuật toán tối ưu hóa Gradient Descent để cập nhật trọng số. 
    *   **Đánh giá:** Mô hình được đánh giá dựa trên các chỉ số Accuracy, Precision, Recall, và F1-Score trên tập kiểm tra (20% dữ liệu).

### **Cài đặt & Môi trường (Installation & Setup)**

Để chạy lại dự án này, hãy làm theo các bước sau:

1.  **Clone repository:**
    ```bash
    git clone https://github.com/banlanhat69/DataSience-CreditCard.git
    cd DataSience-CreditCard
    ```
2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

### **Hướng dẫn sử dụng**

Dự án được chia thành các file Jupyter Notebook, nên chạy theo thứ tự sau:

1.  **`notebooks/01_data_exploration.ipynb`:** Chạy notebook này để xem các bước phân tích, trực quan hóa và rút ra các nhận định ban đầu về dữ liệu.
2.  **`notebooks/02_preprocessing.ipynb`:** Chứa toàn bộ quy trình tiền xử lý dữ liệu, từ lựa chọn đặc trưng đến chuẩn hóa và lưu lại file dữ liệu sạch.
3.  **`notebooks/03_modeling.ipynb`:** Tải dữ liệu đã xử lý, huấn luyện và đánh giá mô hình Logistic Regression.

### **Kết quả**

#### **Kết quả phân tích khám phá**
*   Dữ liệu bị mất cân bằng với 16.1% khách hàng rời bỏ.
*   Các đặc trưng về hành vi giao dịch (`Total_Trans_Ct`, `Total_Trans_Amt`, `Total_Revolving_Bal`) là những yếu tố dự báo mạnh nhất, cho thấy sự khác biệt rõ rệt giữa hai nhóm khách hàng.

#### **Kết quả mô hình**
Mô hình Logistic Regression sau khi được huấn luyện trên bộ dữ liệu đã tối ưu hóa đạt được hiệu suất sau trên tập kiểm tra:

*   **Accuracy:** 89.9%
*   **Precision:** 75.2%
*   **Recall:** 58.6%
*   **F1-Score:** 0.659

**Phân tích:** Mô hình hoạt động tốt trong việc xác định khách hàng trung thành và có độ tin cậy khá khi cảnh báo khách hàng rủi ro (Precision cao). Tuy nhiên, điểm yếu lớn nhất là Recall ở mức trung bình (58.6%), nghĩa là mô hình vẫn còn bỏ sót khoảng 41.4% khách hàng sắp rời đi.

### **Cấu trúc Project**
DataSience-CreditCard/  
├── README.md # File mô tả tổng quan dự án  
├── requirements.txt # Danh sách các thư viện cần thiết  
├── data/  
│ ├── raw/ # Chứa dữ liệu thô ban đầu  
│ └── processed/ # Chứa dữ liệu đã được xử lý và làm sạch  
├── notebooks/  
│ ├── 01_data_exploration.ipynb # Phân tích và khám phá dữ liệu  
│ ├── 02_preprocessing.ipynb # Tiền xử lý dữ liệu  
│ └── 03_modeling.ipynb # Xây dựng và đánh giá mô hình  
├── src/  
│ ├── data_processing.py # Các hàm hỗ trợ tiền xử lý dữ liệu  
│ ├── visualization.py # Các hàm hỗ trợ vẽ biểu đồ  
│ └── models.py # Lớp cài đặt mô hình Logistic Regression  

### **Thách thức & Giải pháp**
*   **Thách thức 1:** Xử lý file CSV có nhiều kiểu dữ liệu khác nhau chỉ bằng NumPy.
    *   **Giải pháp:** Đọc toàn bộ dữ liệu dưới dạng chuỗi (`dtype=str`), sau đó viết các hàm để chuyển đổi từng cột sang kiểu dữ liệu số phù hợp một cách thủ công.
*   **Thách thức 2:** Cài đặt mô hình Logistic Regression gặp lỗi `overflow` khi tính toán hàm `sigmoid`.
    *   **Giải pháp:** Viết lại hàm `sigmoid` ổn định hơn về mặt số học, xử lý riêng các trường hợp đầu vào dương và âm để tránh việc tính `np.exp()` với số mũ quá lớn.
*   **Thách thức 3:** Xử lý outliers lặp đi lặp lại nhưng chúng vẫn xuất hiện.
    *   **Giải pháp:** Hiểu ra rằng mỗi lần xóa outliers, phân phối dữ liệu sẽ thay đổi, dẫn đến các giới hạn IQR mới. Quyết định chỉ thực hiện việc loại bỏ outliers đúng một lần để tránh xóa quá nhiều dữ liệu.

### **Hướng phát triển**

*   **Cải thiện Recall:** Áp dụng các kỹ thuật xử lý dữ liệu mất cân bằng như **SMOTE** (tạo dữ liệu giả cho lớp thiểu số) hoặc sử dụng tham số `class_weight='balanced'` trong mô hình để "trừng phạt" nặng hơn khi dự đoán sai các ca rời bỏ.
*   **Thử nghiệm thuật toán khác:** Sử dụng các mô hình mạnh hơn như **Random Forest** hoặc **XGBoost**, vốn thường cho kết quả tốt hơn trên dữ liệu dạng bảng.
*   **Điều chỉnh ngưỡng quyết định:** Thay vì dùng ngưỡng 0.5 mặc định, có thể tìm một ngưỡng tối ưu hơn để cân bằng giữa Precision và Recall tùy theo mục tiêu kinh doanh.

### **Tác giả**
*   **Tên:** Nguyễn Bá Nam
*   **Contact:** 23122043@student.hcmus.edu.vn
*   **GitHub:** [banlanhat69](https://github.com/banlanhat69)

### **License**
Giấy phép Apache License, ver 2.0.