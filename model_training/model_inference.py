from ultralytics import YOLO

# Load a model
model = YOLO(
    model="./model_outputs/banknote_detection_yolo11x_outputs/weights/best.pt",
    task='detect'
)

# Inference on video
results = model.predict(
    source="C:/Users/ASUS/Downloads/test.mp4",
    # source="C:/Users/ASUS/Downloads/im.jpg",
    # source="./annotated_banknote_dataset/train/images/video8_mp4-0280_jpg.rf.eaded03e8571265206937cf1e8cc7aac.jpg",
    project="./model_predict",
    name="prediction",
    show=True,
    save=True,
    amp=True,
    half=True,
    iou=0.55,
    conf=0.6,
    line_width=2,
    save_crop=False,
    save_txt=False,
    show_labels=True,
    verbose=False,
    device='cuda',
    imgsz=640,
)

for result in results:
    if 0 in result.boxes.cls:
        print("Banknote detected")