import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

sns.set_theme(style="whitegrid")

def plot_attrition_pie_chart(attrition_data: np.ndarray):
    """
    Vẽ biểu đồ tròn thể hiện tỷ lệ khách hàng hiện tại và khách hàng đã rời bỏ.
    """
    labels, counts = np.unique(attrition_data, return_counts=True)
    
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    plt.title('Ratio Between Existing and Attrited Customers', fontsize=16)
    plt.show()

def plot_age_distribution(customer_age: np.ndarray, attrition_data: np.ndarray):
    """
    Vẽ biểu đồ histogram phân phối độ tuổi theo trạng thái khách hàng.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data={'Age': customer_age, 'Attrition': attrition_data}, 
                 x='Age', 
                 hue='Attrition', 
                 kde=True, 
                 multiple="stack") 
    plt.title('Age Distribution by Customer Status', fontsize=16)
    plt.xlabel('Age')
    plt.ylabel('Number of Customers')
    plt.show()

def plot_credit_limit_boxplot(attrition_data: np.ndarray, credit_limit_data: np.ndarray):
    """
    Vẽ biểu đồ hộp so sánh hạn mức tín dụng theo trạng thái khách hàng.
    """
    plt.figure(figsize=(10, 7))
    sns.boxplot(x=attrition_data, y=credit_limit_data)
    plt.title('Credit Limit Distribution by Customer Status', fontsize=16)
    plt.xlabel('Customer Status')
    plt.ylabel('Credit Limit ($)')
    plt.show()

def plot_transaction_scatter(trans_count_data: np.ndarray, trans_amount_data: np.ndarray, 
                             attrition_data: np.ndarray):
    """
    Vẽ biểu đồ phân tán mối quan hệ giữa số lượng và giá trị giao dịch.
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=trans_count_data, 
                    y=trans_amount_data, 
                    hue=attrition_data,
                    alpha=0.6)
    plt.title('Relationship between Transaction Count and Amount', fontsize=16)

    plt.xlabel('Total Transaction Count')
    plt.ylabel('Total Transaction Amount ($)')

    x_ticks = np.arange(0, max(trans_count_data) + 1, 10) 
    plt.xticks(x_ticks, rotation=45) 

    y_ticks = np.arange(0, max(trans_amount_data) + 1, 2500) 
    y_labels = [f'{int(tick/1000)}K' for tick in y_ticks] 
    plt.yticks(y_ticks, y_labels)
    
    plt.show()

def plot_correlation_heatmap(numerical_data: np.ndarray, column_names: List[str]):
    """
    Vẽ biểu đồ nhiệt của ma trận tương quan giữa các biến số.
    """
    correlation_matrix = np.corrcoef(numerical_data, rowvar=False)

    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt='.2f', 
                xticklabels=column_names,
                yticklabels=column_names)
    plt.title('Correlation Matrix between Numerical Variables', fontsize=16)
    plt.show()
