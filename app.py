import streamlit as st
import pandas as pd
import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("🚗 Car Price Predictor")
st.write("Apni car ki details daalo aur price check karo!")

# ==========================================
# 2. DATA LOAD KARNA (Cloud-Proof Method)
# ==========================================
@st.cache_data
def load_data():
    # File ka naam exact hona chahiye
    file_name = 'CAR DETAILS FROM CAR DEKHO.csv'
    
    # Method 1: Direct Path (Best for Cloud)
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        # Method 2: Absolute Path (Backup for Local)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # Error Handling: Agar file nahi mili to ye bataega ki folder mein kya-kya hai
            st.error(f"❌ Error: '{file_name}' file nahi mili!")
            st.warning(f"Current Folder mein ye files hain: {os.listdir(current_dir)}")
            st.stop()
    
    # --- Feature Engineering ---
    # Brand nikalna
    df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
    
    # Age calculate karna
    current_year = datetime.datetime.now().year
    df['car_age'] = current_year - df['year']
    
    return df.drop(['name', 'year'], axis=1)

# Data Load call
df = load_data()

# ==========================================
# 3. MODEL TRAINING
# ==========================================
@st.cache_resource
def train_model(data):
    X = data.drop('selling_price', axis=1)
    y = data['selling_price']

    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])

    model.fit(X, y)
    return model

model = train_model(df)

# ==========================================
# 4. USER INPUTS
# ==========================================
st.write("---")
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", sorted(df['brand'].unique()))
    year = st.number_input("Manufacturing Year", 1990, 2025, 2018)
    km_driven = st.number_input("Kilometers Driven", 0, 500000, 40000)
    transmission = st.selectbox("Transmission", df['transmission'].unique())

with col2:
    fuel = st.selectbox("Fuel Type", df['fuel'].unique())
    seller_type = st.selectbox("Seller Type", df['seller_type'].unique())
    owner = st.selectbox("Owner", df['owner'].unique())

# ==========================================
# 5. PREDICTION
# ==========================================
st.write("---")
if st.button("Predict Price 🚀"):
    car_age = datetime.datetime.now().year - year
    
    input_data = pd.DataFrame({
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'brand': [brand],
        'car_age': [car_age]
    })
    
    try:
        prediction = model.predict(input_data)
        st.success(f"✅ Estimated Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")