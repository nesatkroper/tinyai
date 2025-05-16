import numpy as np
import pickle

# Load the model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example features: [sepal_len, sepal_w, petal_len, petal_w]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example iris

# Normalize (same as training)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(load_iris().data)  # Or save/load the scaler
new_flower = scaler.transform(new_flower)

# Predict
pred = model.predict(new_flower)
class_idx = np.argmax(pred)
classes = ['setosa', 'versicolor', 'virginica']
print(f"Predicted: {classes[class_idx]} ({pred})")