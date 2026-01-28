# model/model_development.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# --- 1. Load the Dataset ---
# Feedback: Stop execution if dataset isn't found
data_path = '../train.csv' # Assumes script is in 'model/' folder and csv is in root
if not os.path.exists(data_path):
    # Fallback for when running from root
    data_path = 'train.csv' 
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Error: 'train.csv' not found. Please download the dataset.")

df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully.")

# --- 2. Data Preprocessing ---
# Select the 6 chosen features + Target
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
target = 'SalePrice'

# Subset data
X = df[features].copy()
y = df[target]

# a. Handling missing values (Fill with median for numerical stability)
X = X.fillna(X.median())

# b. Train-Test Split (Validation Step)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Implement Algorithm ---
# We use Random Forest (handles non-linear relationships well)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# --- 4. Train the Model ---
print("🔄 Training the model...")
model.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
# Feedback: Report regression metrics before saving
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance:")
print(f"   RMSE: ${rmse:,.2f}")
print(f"   R² Score: {r2:.4f}")

# --- 6. Save the Model ---
# Ensure the model folder exists
if not os.path.exists('model'):
    os.makedirs('model')

save_path = 'model/house_price_model.pkl'
joblib.dump(model, save_path)
print(f"💾 Model saved to: {save_path}")