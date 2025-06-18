import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = joblib.load("data.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
joblib.dump(model, "food_model.pkl")
print(f"[✓] Model trained. Accuracy: {accuracy:.2f}")
