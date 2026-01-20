# model_development.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1. Load the dataset
try:
    df = pd.read_csv('train.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'train.csv' was not found in this folder.")
    exit()

# 2. Select ONLY the features your App expects
# These MUST match the inputs in app.py
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
target = 'SalePrice'

# 3. Clean the data
print("⚙️  Processing data...")
X = df[features].copy()
y = df[target]

# Fill missing values with 0 or median to prevent crashing
X = X.fillna(X.median())

# 4. Train the Model
print("🧠 Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save the Model
# Check if 'model' folder exists, if not, create it
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model, 'model/house_price_model.pkl')

print("------------------------------------------------------")
print("🎉 SUCCESS! 'house_price_model.pkl' has been created.")
print("📂 Location: inside the 'model' folder.")
print("------------------------------------------------------")