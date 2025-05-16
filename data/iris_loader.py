from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data():
    iris = load_iris()
    X, y = iris.data, iris.target.reshape(-1, 1)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)
    
    return train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

# def load_data():
#     iris = load_iris()
#     X, y = iris.data, iris.target.reshape(-1, 1)
#     encoder = OneHotEncoder(sparse_output=False)
#     y_encoded = encoder.fit_transform(y)
#     return train_test_split(X, y_encoded, test_size=0.2)
