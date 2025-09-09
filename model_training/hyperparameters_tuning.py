from ultralytics import YOLO
import os


YOLO_MODEL_NAME = "yolo11l"

# Load base model (pretrained)
if not os.path.exists(f"./model_training/base_models/{YOLO_MODEL_NAME}.pt"):
    print(f"Error: {YOLO_MODEL_NAME}.pt not found!")
    exit()

model = YOLO(
    model=f"./model_training/base_models/{YOLO_MODEL_NAME}.pt",
    task="detect",
    verbose=True
)

# Run hyperparameter tuning
results = model.tune(
    data="./annotated_banknote_dataset/data.yaml",  # dataset config
    epochs=20,                 # fewer epochs per trial (saves compute)
    imgsz=768,                 # higher resolution for small banknotes
    batch=0.9,                  # auto batch size
    workers=0,                 # adjust based on CPU cores
    device="cuda",             # "cuda" or "cpu"
    warmup_epochs=3,
    patience=5,               # stop trials early if no improvement
    plots=True,
    project="./model_hyp_tuning_outputs",
    name=f"banknote_detection_{YOLO_MODEL_NAME}_tune",
    exist_ok=True,
    iterations=30,             # number of tuning trials (more = better)
    optimize=True      # allow optimizer param search
)

# Best hyperparameters will be printed + saved in model_outputs
print("Best hyp found:", results.best_params)
