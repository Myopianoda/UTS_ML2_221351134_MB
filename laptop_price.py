import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = load_model('laptop_price_model.h5')

# Load the scaler from the pickle file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app interface
st.title("Laptop Price Prediction")

# Petunjuk untuk input
st.markdown("### Please enter the details of the laptop:")

# Inputan detail spesifikasi oleh user
brand = st.selectbox("Select Brand", ["Acer", "Asus", "Dell", "HP", "Lenovo"])

# RAM Size sebagai pilihan dari selectbox
ram_size = st.selectbox("Select RAM Size (GB)", [4, 8, 16, 32], help="Typical range: 4GB to 32GB")

# Storage Capacity sebagai pilihan dari selectbox
storage_capacity = st.selectbox("Select Storage Capacity (GB)", [256, 512, 1000], help="Typical range: 256GB, 512GB, 1000GB")

# Processor Speed menggunakan number_input untuk input manual (1.5 GHz to 5.0 GHz)
processor_speed = st.number_input("Enter Processor Speed (GHz)", min_value=1.5, max_value=5.0, step=0.5, value=2.5, help="Typical range: 1.5 GHz to 5.0 GHz")

# Screen Size menggunakan number_input untuk input manual (11.0 to 17.0 inches)
screen_size = st.number_input("Enter Screen Size (inches)", min_value=11.0, max_value=17.0, step=0.5, value=15.6, help="Typical range: 11.0 to 17.0 inches")

# Weight menggunakan number_input untuk input manual (2.0 to 5.0 kg)
weight = st.number_input("Enter Weight (kg)", min_value=2.0, max_value=5.0, step=0.5, value=2.5, help="Typical range: 2.0kg to 5.0kg")

# Tombol Proses
if st.button("Predict Price"):
    # One-hot encode untuk brand
    brand_encoded = [1 if brand == b else 0 for b in ["Acer", "Asus", "Dell", "HP", "Lenovo"]]
    new_laptop = np.array([[processor_speed, ram_size, storage_capacity, screen_size, weight] + brand_encoded])

    # Scale the input data using the loaded scaler
    new_laptop_scaled = scaler.transform(new_laptop)

    # Predict the price for the new laptop using the model
    predicted_price_rub = model.predict(new_laptop_scaled)

    # Konversi RUB ke IDR/USD
    RUB_to_USD = 0.012 
    RUB_to_IDR = 198.54
    predicted_price_usd = predicted_price_rub[0][0] * RUB_to_USD
    predicted_price_idr = predicted_price_rub[0][0] * RUB_to_IDR

    # Menampilkan prediksi harga dengan mata uang berbeda
    st.write(f"Predicted Price for the new laptop:")
    st.write(f"Price in Russian Rubles (RUB): ₽{predicted_price_rub[0][0]:,.2f}")
    st.write(f"Price in United States Dollar (USD): ${predicted_price_usd:,.2f}")
    st.write(f"Price in Indonesian Rupiah (IDR): Rp{predicted_price_idr:,.2f}")
