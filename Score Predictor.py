# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset (Hours studied vs Scores)
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Scores': [35, 40, 50, 55, 65, 70, 80, 85]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Hours']]
y = df['Scores']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
hours = np.array([[9]])
predicted_score = model.predict(hours)

print(f"Predicted score for 9 hours study: {predicted_score[0]:.2f}")

# Plot graph
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Study Hours vs Score Prediction")
plt.show()
