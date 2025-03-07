import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Sample dataset: Temperature (°C), Humidity (%), Rainfall (mm), Soil Moisture (%)
data = {
    "Temperature": [25, 30, 20, 35, 28, 22, 33, 26, 29, 31, 27, 21, 24, 32, 23],
    "Humidity": [60, 50, 70, 40, 55, 65, 45, 62, 53, 48, 58, 72, 67, 42, 68],
    "Rainfall": [5, 0, 10, 0, 2, 8, 0, 6, 1, 0, 3, 9, 7, 0, 11],
    "Soil_Moisture": [70, 50, 80, 40, 60, 75, 45, 68, 55, 48, 62, 78, 72, 42, 85]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define Features (X) and Target (y)
X = df[["Temperature", "Humidity", "Rainfall"]]
y = df["Soil_Moisture"]

# Split dataset into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the AI model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate error
error = mean_absolute_error(y_test, y_pred)
print(f"Model Error: {error:.2f}%")

# Function to predict soil moisture and suggest irrigation
def predict_soil_moisture(temp, hum, rain):
    input_data = np.array([[temp, hum, rain]])
    moisture = model.predict(input_data)[0]
    irrigation = "Irrigation Needed" if moisture < 50 else "No Irrigation Needed"
    return moisture, irrigation

# Example Prediction
temp, hum, rain = 27, 58, 3
moisture, irrigation = predict_soil_moisture(temp, hum, rain)

print(f"Predicted Soil Moisture: {moisture:.2f}%")
print(f"Recommendation: {irrigation}")

# Plot actual vs predicted moisture
plt.scatter(y_test, y_pred, color='blue', label="Predicted vs Actual")
plt.xlabel("Actual Soil Moisture (%)")
plt.ylabel("Predicted Soil Moisture (%)")
plt.title("Soil Moisture Prediction Accuracy")
plt.legend()
plt.show()