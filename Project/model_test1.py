import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter
from tensorflow.keras.models import load_model
import os

# Load the model
model = load_model("waste_classification.keras")

# Prediction function
def predict_func(img):
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3]) / 255.0  # Normalize image
    result = np.argmax(model.predict(img))
    if result == 0:
        print("\033[94m" + "This image -> Organic" + "\033[0m")
    elif result == 1:
        print("\033[94m" + "This image -> Recyclable" + "\033[0m")

# Test the model with a sample image
test_image_path = "./DATASET/TEST/Recycling/Image_68(1).png"
if os.path.exists(test_image_path):
    test_img = cv2.imread(test_image_path)
    predict_func(test_img)
else:
    print("Test image not found. Please check the file path.")

####################

# Set up the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam. Please check your system's camera permissions.")
    exit()

cap.set(3, 1240)
cap.set(4, 720)

# Preprocess input for the model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = img_to_array(frame_resized)
    frame_preprocessed = preprocess_input(frame_array)
    return np.expand_dims(frame_preprocessed, axis=0)

# Draw a guide on the frame
def draw_guide(frame):
    height, width, _ = frame.shape
    # Draw a rectangle in the center of the frame
    top_left = (width // 4, height // 4)
    bottom_right = (3 * width // 4, 3 * height // 4)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    # Optionally, add text to indicate where to place the item
    cv2.putText(frame, 'Place item here', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

while True:  # Live object detection
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Draw guide on the frame
    frame = draw_guide(frame)

    # Preprocess the frame for the model
    preprocessed_frame = preprocess_frame(frame)

    # Predict the class
    predictions = model.predict(preprocessed_frame)
    result = np.argmax(predictions)
    if result == 0:
        cv2.putText(frame, "Organic", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    elif result == 1:
        cv2.putText(frame, "Recyclable", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Recycling Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
