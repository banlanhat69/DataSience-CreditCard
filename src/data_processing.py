import numpy as np

def handle_missing_values(numeric_column: np.ndarray, strategy='mean') -> np.ndarray:
    """
    Điền các giá trị thiếu (NaN) trong một cột số bằng mean hoặc median.
    """
    if strategy == 'mean':
        fill_value = np.nanmean(numeric_column)
    elif strategy == 'median':
        fill_value = np.nanmedian(numeric_column)
    else:
        raise ValueError("Strategy must be 'mean' or 'median'")

    missing_indices = np.isnan(numeric_column)
    numeric_column[missing_indices] = fill_value
    return numeric_column

def get_outlier_indices_iqr(numeric_column: np.ndarray) -> np.ndarray:
    """
    Xác định và trả về chỉ số của các hàng chứa outliers bằng phương pháp IQR.
    """
    Q1 = np.percentile(numeric_column, 25)
    Q3 = np.percentile(numeric_column, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_indices = np.where((numeric_column < lower_bound) | (numeric_column > upper_bound))[0]
    
    return outlier_indices

def standardize_features(numeric_data: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa các cột của dữ liệu số bằng Z-score scaling.
    """
    mean = np.mean(numeric_data, axis=0)
    std = np.std(numeric_data, axis=0)

    std[std == 0] = 1 
    
    scaled_data = (numeric_data - mean) / std
    return scaled_data

def create_new_features(data: np.ndarray, header: list):
    """
    Tạo các đặc trưng mới từ dữ liệu hiện có.
    """

    credit_limit_idx = header.index('Credit_Limit')
    revolving_bal_idx = header.index('Total_Revolving_Bal')
    trans_amt_idx = header.index('Total_Trans_Amt')
    trans_ct_idx = header.index('Total_Trans_Ct')
    
    credit_limit = data[:, credit_limit_idx].astype(float)
    revolving_bal = data[:, revolving_bal_idx].astype(float)
    trans_amt = data[:, trans_amt_idx].astype(float)
    trans_ct = data[:, trans_ct_idx].astype(float)

    utilization_ratio = revolving_bal / (credit_limit + 1e-6)

    avg_trans_value = trans_amt / (trans_ct + 1e-6)

    new_features = np.c_[utilization_ratio, avg_trans_value]
    new_data = np.hstack((data, new_features))

    new_header = header + ['Utilization_Ratio', 'Avg_Transaction_Value']
    
    return new_data, new_header

def welch_ttest_numpy(sample1: np.ndarray, sample2: np.ndarray):
    """
    Thực hiện Welch's t-test để so sánh trung bình của hai mẫu độc lập.
    """
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

    numerator = mean1 - mean2
    denominator = np.sqrt(var1 / n1 + var2 / n2)
    t_statistic = numerator / denominator

    df_numerator = (var1 / n1 + var2 / n2)**2
    df_denominator = ((var1 / n1)**2 / (n1 - 1)) + ((var2 / n2)**2 / (n2 - 1))
    degrees_of_freedom = df_numerator / df_denominator
    
    return t_statistic, degrees_of_freedom

def one_hot_encode_column(categorical_column: np.ndarray):
    """
    Thực hiện One-Hot Encoding cho một cột dữ liệu phân loại.

    """

    unique_categories = np.unique(categorical_column)
    num_categories = len(unique_categories)
    integer_encoded = (categorical_column[:, None] == unique_categories).argmax(axis=1)
    one_hot_matrix = np.eye(num_categories, dtype=int)[integer_encoded]
    new_column_names = [f"{unique_categories[i]}" for i in range(num_categories)]
    
    return one_hot_matrix, new_column_names


def frequency_encode_column(categorical_column: np.ndarray) -> np.ndarray:
    """
    Thực hiện Frequency Encoding cho một cột dữ liệu phân loại.
    Thay thế mỗi category bằng tần suất xuất hiện của nó.
    """

    unique_values, counts = np.unique(categorical_column, return_counts=True)
    frequency_map = dict(zip(unique_values, counts))
    mapper = np.vectorize(frequency_map.get)
    frequency_encoded_column = mapper(categorical_column)
    
    return frequency_encoded_column