from utils import predict_food, estimate_calories

# List of test image paths for multiple foods
image_paths = [
    "dataset/apple/apple1.jpg",
    "dataset/banana/banana1.jpg",
    "dataset/burger/burger1.jpg"
]


for img_path in image_paths:
    try:
        predicted_food = predict_food(img_path)
        calories = estimate_calories(predicted_food)

        print(f"\nImage: {img_path}")
        if calories:
            print(f"Food: {predicted_food}\nEstimated Calories: {calories:.2f} kcal")
        else:
            print(f"Food: {predicted_food}\nCalories info not found in database.")
    except Exception as e:
        print(f"\nImage: {img_path}")
        print(f"[!] Error: {e}")
