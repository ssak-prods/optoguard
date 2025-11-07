import cv2
import time
from detector import ObjectDetector
from speaker import Speaker
from utils import DetectionManager
from watchdog import WatchdogMonitor, AlertLevel

def main():
    # Initialize components
    detector = ObjectDetector(confidence_threshold=0.5)
    speaker = Speaker(rate=150, volume=1.0)
    detection_manager = DetectionManager(cooldown_seconds=5.0)
    watchdog = WatchdogMonitor(
        cooldown_seconds=5.0,
        empty_scene_cooldown=30.0,
        min_confidence=0.5
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("AI Object Detector is running!")
    print("Show a bottle to toggle Watchdog Mode")
    print("Press 'q' to quit")
    
    # State variables
    watchdog_mode = False
    monitoring_mode = True
    last_mode_switch = 0
    mode_switch_cooldown = 2.0  # seconds
    bottle_detected = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            detections = detector.detect(frame)
            
            # Check for bottle to toggle mode
            current_time = time.time()
            bottle_in_frame = any(name == 'bottle' for name, _, _ in detections)
            
            # Mode switching logic
            if bottle_in_frame and not bottle_detected and (current_time - last_mode_switch) > mode_switch_cooldown:
                watchdog_mode = not watchdog_mode
                last_mode_switch = current_time
                mode_msg = "Watchdog Mode enabled" if watchdog_mode else "Watchdog Mode disabled"
                print(mode_msg)
                speaker.speak(mode_msg)
            
            bottle_detected = bottle_in_frame
            
            if monitoring_mode:
                if watchdog_mode:
                    # Process scene with watchdog
                    alerts = watchdog.process_scene(detections)
                    for alert_level, message in alerts:
                        print(f"[{alert_level.name}] {message}")
                        speaker.speak(message)
                else:
                    # Normal mode - use spatial description
                    description = detection_manager.get_spatial_description(detections)
                    if description:
                        print(f"🔊 Speaking: \"{description}\"")
                        speaker.speak(description)
            
            # Draw bounding boxes and labels
            for name, conf, bbox in detections:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), \
                                int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                
                # Draw box with different colors for bottle
                color = (0, 0, 255) if name == 'bottle' else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add mode indicator
            mode_text = "Watchdog Mode" if watchdog_mode else "Normal Mode"
            cv2.putText(frame, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(frame, "Show bottle to toggle mode", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('AI Object Detector', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to prevent high CPU usage
            #time.sleep(0.01)
            
    finally:
        cap.release() 
        cv2.destroyAllWindows()
        print("AI Object Detector has been stopped.")

if __name__ == "__main__":
    main() 