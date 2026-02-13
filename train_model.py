from ultralytics import YOLO

# Path to dataset
DATASET_PATH = "dataset/train_split"

# Load YOLOv8 classification model
model = YOLO("yolov8s-cls.pt")

# Train model
model.train(
    data=DATASET_PATH,
    epochs=5
,
    imgsz=224,
    batch=16,
    name="crack_classifier",
    project="runs"
)

print("Training completed successfully.")
