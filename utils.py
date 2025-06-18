import cv2
import joblib
import numpy as np
import pandas as pd

IMG_SIZE = (100, 100)
cal_df = pd.read_csv("calories.csv")
model = joblib.load("food_model.pkl")
le = joblib.load("label_encoder.pkl")

def predict_food(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE).flatten().reshape(1, -1)
    pred = model.predict(img)[0]
    label = le.inverse_transform([pred])[0]
    return label

def estimate_calories(food, volume_cm3=100):  # Assume volume if not known
    row = cal_df[cal_df['food'] == food]
    if row.empty:
        return None
    density = row.iloc[0]['density_g_cm3']
    cal_per_gram = row.iloc[0]['calories_per_gram']
    mass = density * volume_cm3
    return mass * cal_per_gram
