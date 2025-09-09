from ultralytics import YOLO

# Load a model
model = YOLO(
    model="./model_outputs/banknote_detection_yolo12l_resume_outputs/weights/best.pt",
    task='detect'
)


def validate_model(model: YOLO):
    metrics = model.val(
        data="./annotated_banknote_dataset/data.yaml",
        project='./model_validation_outputs',
        split='test',
        name='test'
    )
        
    return metrics

if __name__ == '__main__':
    metrics = validate_model(model)
    print(metrics)
