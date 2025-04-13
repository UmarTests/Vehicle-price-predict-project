import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import joblib

# ======================= PHASE 1: DATA LOADING & CLEANING =======================
print("\n🚗 Loading Dataset...")
df = pd.read_csv(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\vehicle_pri.pro\Vehicle Price Prediction\dataset.csv")
print("✅ Dataset Loaded")

# Drop missing target values
df = df.dropna(subset=["price"])

# Fill missing numerical values with median
num_cols = ["cylinders", "mileage", "doors"]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill missing categorical values with "Unknown"
cat_cols = ["engine", "fuel", "transmission", "trim", "body", "exterior_color", "interior_color"]
df[cat_cols] = df[cat_cols].fillna("Unknown")

# Drop duplicates
df = df.drop_duplicates()

print("✅ Data Cleaning Done")

# ======================= PHASE 2: EDA & VISUALIZATION =======================
print("\n📊 Performing EDA...")

# Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=30, kde=True, color="blue")
plt.title("Distribution of Vehicle Prices")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# ploting a bee swarm plot
plt.figure(figsize=(8, 5))
sns.stripplot(x=df["fuel"], y=df["price"], palette="Set2", jitter=True)
plt.xlabel("Fuel Type")
plt.ylabel("Price (USD)")
plt.title("Vehicle Price by Fuel Type")
plt.show()


print("✅ EDA Completed")

# ======================= PHASE 3: FEATURE ENGINEERING =======================
print("\n⚙️ Feature Engineering...")

# Standardizing fuel types
df['fuel'] = df['fuel'].replace({'E85 Flex Fuel': 'Flex Fuel', 'PHEV Hybrid Fuel': 'Hybrid'})

# One-Hot Encoding
df = pd.get_dummies(df, columns=['fuel', 'transmission'], drop_first=True)

# Label Encoding for Model & Color
encoder = LabelEncoder()
df['model'] = encoder.fit_transform(df['model'])
df['exterior_color'] = encoder.fit_transform(df['exterior_color'])

# Log Transform Target Variable
df["log_price"] = np.log1p(df["price"])

print("✅ Feature Engineering Done")

# ======================= PHASE 4: MODEL TRAINING =======================
print("\n🎯 Training Model...")
df = df.drop(columns=['name', 'description'], errors='ignore')

from sklearn.preprocessing import LabelEncoder

categorical_cols = ["make", "engine", "trim", "body", "interior_color", "drivetrain"]
encoder = LabelEncoder()
print(df.dtypes)

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Feature Selection (Top 30 most important features)
X = df.drop(columns=["price", "log_price"])
y = df["log_price"]


# Encode categorical features before train-test split
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# Define categorical features that need encoding
categorical_features = ["make", "engine", "trim", "body", "interior_color", "drivetrain"]

# Apply label encoding
for col in categorical_features:
    df[col] = encoder.fit_transform(df[col])

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm no object columns exist
print(X_train.dtypes)

# Train LightGBM Model
model = LGBMRegressor(n_estimators=500, max_depth=15, num_leaves=50, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

print("✅ Model Training Completed")

# ======================= PHASE 5: MODEL EVALUATION =======================
print("\n📈 Evaluating Model...")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict and convert back from log scale
y_pred = np.expm1(model.predict(X_test))
y_test_original = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
mae = mean_absolute_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 MAE: {mae:.2f}")
print(f"🔹 R²: {r2:.2f}")

# ======================= PHASE 6: SAVE MODEL =======================
print("\n💾 Saving Model...")
joblib.dump(model, "vehicle_price_model.joblib")
joblib.dump(encoder, "encoder.joblib")
joblib.dump(X_train.columns.tolist(), "features_list.joblib")

print("✅ Model & Preprocessor Saved!")

import joblib
encoder = joblib.load("encoder.joblib")
print(type(encoder))
