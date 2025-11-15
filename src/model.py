import numpy as np

class MyLogisticRegression:
    """
    Cài đặt mô hình Logistic Regression.
    """
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Hàm Sigmoid
        """
        z = np.asarray(z)
        result = np.zeros_like(z, dtype=float)
        positive_z_mask = (z >= 0)
        result[positive_z_mask] = 1 / (1 + np.exp(-z[positive_z_mask]))
        negative_z_mask = (z < 0)
        result[negative_z_mask] = np.exp(z[negative_z_mask]) / (1 + np.exp(z[negative_z_mask]))
        
        return result

    def fit(self, X, y):
        """
        Huấn luyện mô hình bằng thuật toán Gradient Descent.
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):

            linear_model = np.dot(X, self.weights) + self.bias

            y_predicted_prob = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted_prob - y))
            db = (1 / n_samples) * np.sum(y_predicted_prob - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Dự đoán xác suất.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Dự đoán nhãn (0 hoặc 1) dựa trên một ngưỡng.
        """
        y_predicted_prob = self.predict_proba(X)
        y_predicted_class = [1 if i > threshold else 0 for i in y_predicted_prob]
        return np.array(y_predicted_class)