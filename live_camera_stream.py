from re import I
import cv2
import numpy as np
from ultralytics import YOLO
import time

def load_yolo_model(
    # model_path='./model_outputs/banknote_detection_yolo11x_outputs/weights/best.pt'
    model_path='./exported_models/yolo11x_banknote/tensorrt/yolo11x_banknote.engine'
    ):
    """
    Load YOLO model
    Args:
        model_path (str): Path to YOLO model file
    Returns:
        YOLO model object
    """
    try:
        model = YOLO(model_path)
        print(f"✅ YOLO model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return None

def draw_detections(frame, results, confidence_threshold):
    """
    Draw bounding boxes and labels on frame
    Args:
        frame: OpenCV frame
        results: YOLO detection results
        confidence_threshold: Minimum confidence for detection
    Returns:
        Frame with drawn detections
    """
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        print(results[0].names)
        for box in boxes:
            # Get confidence score
            confidence = box.conf[0].cpu().numpy()
            
            if confidence > confidence_threshold:
                # print(f"Confidence: {confidence_threshold}")
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class ID and name
                class_id = int(box.cls[0].cpu().numpy())
                class_name = results[0].names[class_id]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label with class name and confidence
                label = f"{class_name}: {confidence:.2f}"
                
                # Get label size for background rectangle
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_height - 10), 
                    (x1 + label_width, y1), 
                    (0, 255, 0), 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 0), 
                    2
                )
    
    return frame

def stream_rtsp_with_yolo(url, confidence_threshold):
    """
    Stream RTSP video with YOLO object detection
    Args:
        url (str): RTSP URL to stream
        confidence_threshold (float): Minimum confidence for detections
    """
    print(f"Connecting to: {url}")
    print("Loading YOLO model...")
    
    # Load YOLO model
    model = load_yolo_model()
    if model is None:
        return False
    
    print("Press 'q' to quit the stream")
    
    # Create VideoCapture object
    cap = cv2.VideoCapture(url)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera stream")
        return False
    
    print("✅ Connected! Streaming live video with YOLO detection...")
    print("Press 'q' in the video window to quit")
    print("Press 'c' to toggle confidence threshold display")
    
    # Variables for FPS calculation
    fps_counter = 0
    start_time = time.time()
    show_fps = True
    show_confidence = True
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if ret:
            # Resize frame to make window smaller and improve performance
            small_frame = cv2.resize(frame, (640, 640))
            
            # Run YOLO inference
            try:
                results = model(
                    small_frame,
                    verbose=False,
                    # device='cuda',
                    imgsz=640,
                    half=True,
                    conf=confidence_threshold,
                    iou=0.55,
                    amp=True,
                )
                
                # Draw detections on frame
                annotated_frame = draw_detections(
                    frame=small_frame, 
                    results=results, 
                    confidence_threshold=confidence_threshold
                )
                
            except Exception as e:
                print(f"Error during inference: {e}")
                annotated_frame = small_frame
            
            # Calculate and display FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= 1.0:
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()
                
                if show_fps:
                    cv2.putText(
                        annotated_frame, 
                        f"FPS: {fps:.1f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 255), 
                        2
                    )
            
            # Display confidence threshold info
            if show_confidence:
                cv2.putText(
                    annotated_frame, 
                    f"Confidence: {confidence_threshold:.2f}", 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 0), 
                    1
                )
            
            # Display the frame with detections
            cv2.imshow('RTSP Live Stream with YOLO', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_confidence = not show_confidence
                print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"Confidence threshold: {confidence_threshold:.2f}")
            elif key == ord('-'):
                confidence_threshold = max(0.05, confidence_threshold - 0.05)
                print(f"Confidence threshold: {confidence_threshold:.2f}")
                
        else:
            print("Error: Could not read frame")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Stream ended")
    return True

def main():
    # Configuration
    rtsp_url = 0  # Change to your RTSP URL or camera index
    confidence_threshold = 0.6
    
    print("=== RTSP Live Stream with YOLO Object Detection ===\n")
    print("Controls:")
    print("  'q' - Quit")
    print("  'c' - Toggle confidence display")
    print("  '+' - Increase confidence threshold")
    print("  '-' - Decrease confidence threshold")
    print()
    
    # Start streaming with YOLO detection
    stream_rtsp_with_yolo(rtsp_url, confidence_threshold)

if __name__ == "__main__":
    main()