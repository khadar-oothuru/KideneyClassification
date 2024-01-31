import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('trainedModele.h5')

# Define the input size expected by the model
EXPECTED_INPUT_SIZE = (64, 64)

# Load and preprocess a test image
test_image_path = 'datasets/Normal/Normal- (105).jpg'
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, EXPECTED_INPUT_SIZE)
test_image = np.reshape(test_image, [1, *EXPECTED_INPUT_SIZE, 3])  # Reshape to (1, 64, 64, 3)
test_image = test_image / 255.0  # Normalize pixel values between 0 and 1

# Make predictions
predictions = model.predict(test_image)

# Get the predicted class
predicted_class = np.argmax(predictions)

# Map the predicted class to the corresponding label
class_labels = {0: 'Normal', 1: 'Tumor', 2: 'Cyst', 3: 'Stone'}
predicted_label = class_labels[predicted_class]

# Print the result
print(f'The model predicts that the image belongs to class: {predicted_label}')
