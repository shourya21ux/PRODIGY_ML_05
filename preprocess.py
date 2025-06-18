import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

IMG_SIZE = (100, 100)

def load_images_from_folder(folder):
    X, y = [], []
    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE).flatten()
                    X.append(img)
                    y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_images_from_folder("dataset")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    joblib.dump((X, y_encoded), "data.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("[✓] Preprocessing done. Saved features and labels.")
