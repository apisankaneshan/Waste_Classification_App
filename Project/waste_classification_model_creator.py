import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import cv2
import os
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Paths for training and testing data
train_path = './DATASET/TRAIN'
test_path = './DATASET/TEST'


# Load and preprocess data for visualization
x_data = []
y_data = []

for category in glob(train_path + '/*'):
    for file in tqdm(glob(category + '/*')):
        img_array = cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split("/")[-1])

data = pd.DataFrame({'image': x_data, 'label': y_data})

# Data visualization
colors = ['#a0d157', '#c48bb8']
plt.pie(data.label.value_counts(), startangle=90, explode=[0.05, 0.05], autopct='%0.2f%%',
        labels=['Organic', 'Recyclable'], colors=colors, radius=2)
plt.show()

plt.figure(figsize=(20, 15))
for i in range(9):
    plt.subplot(4, 3, (i % 12) + 1)
    index = np.random.randint(len(data))
    plt.title('This image is of {0}'.format(data.label[index]), fontdict={'size': 20, 'weight': 'bold'})
    plt.imshow(data.image[index])
    plt.tight_layout()

# Count number of classes
className = glob(train_path + '/*')
numberOfClass = len(className)
print("Number Of Class: ", numberOfClass)

# Define ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode="rgb",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    color_mode="rgb",
    class_mode="categorical"
)

# Implementing transfer learning with MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(numberOfClass, activation='softmax')(x)

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
hist = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Plot accuracy and loss
plt.figure(figsize=[10, 6])
plt.plot(hist.history["accuracy"], label="Train acc")
plt.plot(hist.history["val_accuracy"], label="Validation acc")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], label="Train loss")
plt.plot(hist.history['val_loss'], label="Validation loss")
plt.legend()
plt.show()

# Prediction function
def predict_func(img):
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3]) / 255.0  # Normalize image
    result = np.argmax(model.predict(img))
    if result == 0:
        print("\033[94m" + "This image -> Recyclable" + "\033[0m")
    elif result == 1:
        print("\033[94m" + "This image -> Organic" + "\033[0m")

# Test the model with a sample image
test_img = cv2.imread("./DATASET/TEST/Recycling/Image_68(1).png")
predict_func(test_img)

model.save("waste_classification.keras")