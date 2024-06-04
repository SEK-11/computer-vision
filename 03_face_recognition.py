import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('/home/vmukti/Desktop/try2/OpenCV-Face-Recognition-master/FacialRecognition/runs/detect/train/weights/best.pt')  # Replace with the path to your trained YOLOv8 model

# Access the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 face detection
    results = model(frame)

    # Draw bounding boxes around detected faces
    for result in results:
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            confidence = result.boxes.conf[i].item()
            class_id = result.boxes.cls[i].item()
            label = "Shantanu" if class_id == 0 else "Unknown"

            # Draw the bounding box
            color = (0, 255, 0) if label == "Shantanu" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Put the label on the bounding box
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
