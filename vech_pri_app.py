import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --------------------------------------
# 🚀 Load Model, Features List, and Encoder
# --------------------------------------
# Update these paths to match where your files are saved.
model_path = r"C:\Users\mohdq\OneDrive\Desktop\internship projects\vehicle_pri.pro\Vehicle Price Prediction\vehicle_price_model.joblib"
features_list_path = r"C:\Users\mohdq\OneDrive\Desktop\internship projects\vehicle_pri.pro\Vehicle Price Prediction\features_list.joblib"
encoder_path = r"C:\Users\mohdq\OneDrive\Desktop\internship projects\vehicle_pri.pro\Vehicle Price Prediction\encoder.joblib"

try:
    model = joblib.load(model_path)
    expected_features = joblib.load(features_list_path)
    encoder = joblib.load(encoder_path)  # (May be used for debugging or future extension)
    st.success("✅ Model, Features List, and Encoder loaded successfully!")
except Exception as e:
    st.error(f"🚨 Error loading files: {e}")
    st.stop()

# --------------------------------------
# 🎨 Streamlit UI Setup
# --------------------------------------
st.title("🚗 Vehicle Price Prediction App")
st.markdown("Enter the vehicle details below to get an estimated price. All inputs are user friendly.")

# Dropdown options – these must match the training categories.
years = list(range(1990, 2025))
mileage_options = [10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]
cylinders_options = [3, 4, 6, 8, 10, 12]
doors_options = [2, 3, 4, 5, 6]

# Categorical options as used during training:
car_models = ["Toyota Camry", "Honda Accord", "Ford Focus", "BMW 3 Series", "Mercedes C Class"]
fuel_types = ["Gasoline", "Diesel", "Electric", "Hybrid", "Flex Fuel", "Other"]
transmissions = ["Manual", "Automatic", "CVT", "Other"]
exterior_colors = ["White", "Black", "Blue", "Red", "Silver", "Other"]

# --------------------------------------
# 🛠️ Collect User Inputs
# --------------------------------------
year = st.selectbox("📅 Year of Manufacture", years, index=years.index(2020))
mileage = st.selectbox("📏 Total Mileage (km driven)", mileage_options, index=mileage_options.index(50000))
cylinders = st.selectbox("🔧 Number of Cylinders", cylinders_options, index=cylinders_options.index(4))
doors = st.selectbox("🚪 Number of Doors", doors_options, index=doors_options.index(4))

selected_model = st.selectbox("🚘 Select Car Model", car_models, index=0)
selected_fuel = st.selectbox("⛽ Select Fuel Type", fuel_types, index=0)
selected_transmission = st.selectbox("⚙️ Select Transmission", transmissions, index=0)
selected_color = st.selectbox("🎨 Select Exterior Color", exterior_colors, index=0)

# --------------------------------------
# 🔄 Convert User Inputs to Model Format
# --------------------------------------
# Create mapping dictionaries for categorical inputs.
model_mapping = {m: i for i, m in enumerate(car_models)}
color_mapping = {c: i for i, c in enumerate(exterior_colors)}

# For fuel and transmission, we'll use one-hot encoding.
fuel_one_hot = {f"fuel_{f}": 0 for f in fuel_types}
transmission_one_hot = {f"transmission_{t}": 0 for t in transmissions}

# Set the selected fuel and transmission to 1.
fuel_one_hot[f"fuel_{selected_fuel}"] = 1
transmission_one_hot[f"transmission_{selected_transmission}"] = 1

# Convert selected car model and exterior color using mapping dictionaries.
mapped_model = model_mapping[selected_model]
mapped_color = color_mapping[selected_color]

# --------------------------------------
# 📝 Create Raw Input DataFrame
# --------------------------------------
# Note: We're using only numeric values here.
raw_input = pd.DataFrame([{
    "year": year,
    "mileage": mileage,
    "cylinders": cylinders,
    "doors": doors,
    "model": mapped_model,
    "exterior_color": mapped_color,
    # Merge one-hot encoded fuel and transmission values:
    **fuel_one_hot,
    **transmission_one_hot
}])

st.write("### Raw Input Data:", raw_input)

# --------------------------------------
# 🔄 Align Input Data with Expected Features
# --------------------------------------
# Reindex the DataFrame to match the features used during training.
if expected_features is not None:
    final_input = raw_input.reindex(columns=expected_features, fill_value=0)
else:
    final_input = raw_input

st.write("### Final Input Data (Ordered):", final_input)

# --------------------------------------
# 🔮 Make Prediction
# --------------------------------------
if st.button("🔍 Predict Price"):
    # Predict (assuming the model was trained on log(price) and needs conversion)
    pred_log_price = model.predict(final_input)[0]
    pred_price = np.expm1(pred_log_price)
    st.success(f"💰 Estimated Vehicle Price: ${pred_price:,.2f}")

# --------------------------------------
# 🚀 Quick Test Runs (For Manual Check)
# --------------------------------------
if __name__ == "__main__":
    test_cases = [
        {"year": 2020, "mileage": 50000, "cylinders": 4, "doors": 4, "model": "Toyota Camry", "fuel": "Gasoline", "transmission": "Automatic", "color": "White"},
        {"year": 2018, "mileage": 75000, "cylinders": 6, "doors": 4, "model": "BMW 3 Series", "fuel": "Diesel", "transmission": "Manual", "color": "Black"},
        {"year": 2015, "mileage": 120000, "cylinders": 4, "doors": 4, "model": "Ford Focus", "fuel": "Hybrid", "transmission": "CVT", "color": "Blue"},
    ]

    for i, test in enumerate(test_cases):
        print(f"\n🔍 Test Case {i+1}: {test}")

        # Convert to model format
        test_input = pd.DataFrame([{
            "year": test["year"],
            "mileage": test["mileage"],
            "cylinders": test["cylinders"],
            "doors": test["doors"],
            "model": model_mapping[test["model"]],
            "exterior_color": color_mapping[test["color"]],
            **{f"fuel_{test['fuel']}": 1 if f"fuel_{test['fuel']}" in expected_features else 0 for f in fuel_types},
            **{f"transmission_{test['transmission']}": 1 if f"transmission_{test['transmission']}" in expected_features else 0 for f in transmissions},
        }]).reindex(columns=expected_features, fill_value=0)

        # Predict
        pred_log_price = model.predict(test_input)[0]
        pred_price = np.expm1(pred_log_price)  # Convert from log scale

        print(f"💰 Predicted Price: ₹{pred_price:,.2f}")
