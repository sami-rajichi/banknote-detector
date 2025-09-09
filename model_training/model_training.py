from ultralytics import YOLO

YOLO_MODEL_NAME = "yolo11x"

def train_banknotes():
    """
    Optimized YOLOv12l training for banknotes detection:
    - Handles tiny/far and close/clear banknotes
    - Multi-scale disabled for faster training
    - Uses AdamW optimizer
    """
    # Load pretrained model
    model = YOLO(f"./model_training/base_models/{YOLO_MODEL_NAME}.pt", task="detect")

    results = model.train(
        # Core setup
        data="./annotated_banknote_dataset/data.yaml",
        epochs=100,
        patience=8,                # Allow longer early stopping
        batch=0.9,                  # Adjust to GPU memory
        imgsz=640,                   # Input resolution
        cache='disk',                # Faster data loading
        device="cuda",
        workers=6,
        project="./model_outputs",
        name=f"banknote_detection_{YOLO_MODEL_NAME}_outputs",
        exist_ok=True,
        pretrained=f"./model_training/base_models/{YOLO_MODEL_NAME}.pt",
        optimizer="AdamW",
        seed=42,
        # single_cls=True,             # Single-class detection
        multi_scale=False,           # Disabled for speed
        resume=False,
        plots=True,
        # amp=True,

        # Freeze backbone partially
        freeze=12 if '12' in YOLO_MODEL_NAME else 10,                    # Fewer frozen layers to help tiny objects
        dropout=0.1,                 # Optional: disable dropout for stable training

        # Learning rate
        lr0=0.001,                    # Initial LR
        lrf=0.05,                     # Final LR fraction
        warmup_epochs=3,
        momentum=0.937,
        weight_decay=0.008,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights
        box=9.0,                      # Higher box loss for precise localization
        cls=0.25,                      # Single class → low classification weight
        dfl=2,                      # Fine-grained box regression

        # Augmentations (optimized for small banknotes)
        hsv_h=0.015,                  # Reduced - preserve banknote colors
        hsv_s=0.5,                   # Moderate saturation changes
        hsv_v=0.3,                   # Moderate brightness changes
        degrees=8.0,                 # Moderate rotation
        translate=0.15,              # Increased - simulate camera movement
        scale=0.8,                   # Wide scale range for size variation
        shear=1.5,                   # Reduced shear (banknotes are rectangular)
        perspective=0.0008,          # Increased perspective for depth
        flipud=0.0,                  # No vertical flip for banknotes
        fliplr=0.5,                  # Horizontal flip OK
        bgr=0.0,                     # No BGR swap

        # Critical augmentations for small objects
        mosaic=0.5,                  # High mosaic for multi-scale learning
        mixup=0.2,                  # Moderate mixup
        # copy_paste=0.2,              # High copy-paste for small objects
        
        # Advanced augmentations
        # auto_augment="randaugment",  # Additional augmentation policy
        erasing=0.2,                 # Reduced random erasing
        
        # Training schedule
        # close_mosaic=15,             # Disable mosaic in last 15 epochs
    )

    return results


def tiny_object_boost_resume(previous_weights):
    """
    Fine-tune for tiny/far banknotes:
    - Balanced loss weights (not too extreme)
    - Moderate augmentations (prevent overfitting + realism)
    - Larger imgsz for detail
    """

    model = YOLO(previous_weights, task="detect")

    results = model.train(
        data="./annotated_banknote_dataset/data.yaml",
        pretrained=previous_weights,
        epochs=50,                     # Short fine-tuning phase
        patience=10,                   # Allow more time before stopping
        batch=0.9,
        imgsz=860,                     # Higher resolution helps small objects
        cache="disk",
        device="cuda",
        workers=6,
        project="./model_outputs",
        name=f"banknote_detection_{YOLO_MODEL_NAME}_resume_outputs",
        exist_ok=True,
        amp=True,
        plots=True,

        # Train more layers (no hard freezing)
        freeze=10 if '12' in YOLO_MODEL_NAME else 8,

        # Optimizer
        optimizer="AdamW",
        lr0=0.002,                     # Lower LR for stability
        lrf=0.05,
        warmup_epochs=3,
        momentum=0.937,
        weight_decay=0.006,            # Reduced → avoids over-regularizing
        dropout=0.1,                   # Small dropout only

        # Balanced loss weights
        box=8.5,                       # Still strong for precise localization
        cls=0.25,                      # Low since single-class
        dfl=2.0,                       # Fine-grained box regression, but stable

        # Safer augmentations
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.25,
        degrees=6.0,                   # Mild rotation
        translate=0.15,
        scale=0.85,
        shear=0.8,
        perspective=0.0008,
        flipud=0.0,
        fliplr=0.5,

        # Augmentations tuned down a bit
        mosaic=0.6,                    # Still strong, but less than 0.8
        mixup=0.1,
        erasing=0.1,
    )

    return results

if __name__ == "__main__":
    print("🚀 Starting YOLO12l Banknote Detection Training")
    print("=" * 50)
    
    # Stage 1: Main training
    print("📈 Stage 1: Main Multi-Scale Training")
    results = train_banknotes()
    print(f"Training completed! Best mAP: {results.best_fitness:.3f}")
    
    # Stage 2: Tiny object boost (optional)
    # use_tiny_boost = input("\\n🔍 Run tiny object boost training? (y/n): ").lower() == 'y'
    
    # if use_tiny_boost:
    #     print("\\n🎯 Stage 2: Tiny Object Boost Training")
    #     previous_weights = f"./model_outputs/banknote_detection_{YOLO_MODEL_NAME}_outputs/weights/best.pt"
        
    #     try:
    #         boost_results = tiny_object_boost_resume(previous_weights)
    #         print(f"Boost training completed! Best mAP: {boost_results.best_fitness:.3f}")
                        
    #     except FileNotFoundError:
    #         print(f"❌ Could not find weights at {previous_weights}")
    #         print("Please check the path or run main training first")
    
    print("\\n✅ Training pipeline completed!")