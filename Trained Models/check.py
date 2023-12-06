import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Path to your trained model and test image
model_path = 'best.pt'
image_path = 'img7.jpg'  # Replace with your image path

# Load the YOLO model
model = YOLO(model_path)

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found or path is incorrect")

# Perform inference
results = model(image)[0]

# Process results
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, results.names[int(class_id)], (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(image_rgb)
plt.title('Results')
plt.show()
