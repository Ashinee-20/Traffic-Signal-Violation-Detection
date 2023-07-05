import cv2
import numpy as np
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Read and process images
input_folder = "input_images"
output_folder = "output_images"
image_names = os.listdir(input_folder)

# Adjust confidence and non-maximum suppression thresholds
confidence_threshold = 0.5
nms_threshold = 0.3

for image_name in image_names:
    image_path = os.path.join(input_folder, image_name)
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Perform preprocessing
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Forward pass through the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detection results
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Vehicle detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Save the detected objects as separate images (excluding traffic and person)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]

            # Exclude saving if the object is "traffic" or "person"
            if label == "traffic light" or label == "person":
                continue

            # Extract the object from the image
            object_img = img[y:y+h, x:x+w]

            # Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Generate a unique filename for the object image
            object_filename = os.path.join(output_folder, f"{label}_{confidence:.2f}_{image_name}_{i}.jpg")

            # Save the object image
            cv2.imwrite(object_filename, object_img)

print("Object detection completed.")
