import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Step 1: Load the dataset
data = pd.read_csv(r'C:\Users\Deekshith Patel L L\OneDrive\Desktop\Dsp Project\server\data\crop_data.csv')

# Step 2: Preprocessing
label_encoder_place = LabelEncoder()
label_encoder_crop = LabelEncoder()

data['Place'] = label_encoder_place.fit_transform(data['Place'])
data['Crop'] = label_encoder_crop.fit_transform(data['Crop'])

# Step 3: Split the data into features (X) and target (y)
X = data.drop(columns=['Crop Price (per Quintal)'])  # Features
y = data['Crop Price (per Quintal)']  # Target variable

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Step 6: Train the model
rf_model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")

# Step 9: Ensure the directory exists before saving the model
model_dir = 'server/model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
joblib.dump(rf_model, os.path.join(model_dir, 'crop_data.pkl'))

# Optionally, visualize feature importances
feature_importance = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)