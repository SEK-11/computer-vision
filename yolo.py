from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8m.pt')  # Using a medium variant of YOLOv8

# Fine-tune the model on your dataset
model.train(data='/home/vmukti/Desktop/try2/OpenCV-Face-Recognition-master/FacialRecognition/Robo_data_2.yolov8/data.yaml', epochs=350, batch=16, imgsz=640)
