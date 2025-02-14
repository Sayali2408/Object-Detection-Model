import cv2
import numpy as np
import mediapipe as mp

# Add these constants at the top of the file
GENDER_MODEL = 'Object detection\gender_net.caffemodel'
GENDER_PROTO = 'Object detection\gender_deploy.prototxt'
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_CONFIDENCE_THRESHOLD = 0.8

# ------------------------ Object Detection (YOLO) ------------------------
class YOLOObjectDetector:
    def __init__(self, weights_path="Object detection\yolov3.weights", config_path="Object detection\yolov3.cfg", names_path="Object detection\coco.names"):
        self.yolo = cv2.dnn.readNet(weights_path, config_path)
        with open(names_path, "r") as file:
            self.classes = [line.strip() for line in file.readlines()]
        layer_names = self.yolo.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.yolo.getUnconnectedOutLayers().flatten()]
        self.colorRed = (0, 0, 255)
        self.colorGreen = (0, 255, 0)
        self.gender_classifier = None  # Will be set from main
    
    def set_gender_classifier(self, classifier):
        self.gender_classifier = classifier
    
    def detect_objects(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)
        
        class_ids, confidences, boxes = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (
                        int(detection[0] * width), int(detection[1] * height),
                        int(detection[2] * width), int(detection[3] * height)
                    )
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                
                # If it's a person, try to detect gender
                if label == "person" and self.gender_classifier and self.gender_classifier.model_loaded:
                    face_img = frame[max(0, y):min(y+h, height), max(0, x):min(x+w, width)].copy()
                    if face_img.size > 0:
                        try:
                            gender = self.gender_classifier.predict_gender(face_img)
                            label = f"person ({gender})"
                        except Exception:
                            pass
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.colorGreen, 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, self.colorRed, 2)

# ------------------------ Hand Gesture Recognition (MediaPipe) ------------------------
class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,  # Increased confidence threshold
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hand_gestures(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        height, width, _ = frame.shape
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.identify_gesture(hand_landmarks)
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                gesture_x, gesture_y = int(wrist.x * width), int(wrist.y * height)
                cv2.putText(frame, gesture, (gesture_x, gesture_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    def identify_gesture(self, hand_landmarks):
        # Get all relevant landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        def distance(p1, p2):
            return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

        def is_finger_extended(tip, pip):
            return tip.y < pip.y - 0.04  # Increased threshold for more definitive extension

        # Calculate palm size for normalization
        palm_size = distance(index_mcp, pinky_pip)

        # Check finger states
        fingers = [
            thumb_tip.x > thumb_ip.x if wrist.x < thumb_mcp.x else thumb_tip.x < thumb_ip.x,  # Thumb
            is_finger_extended(index_tip, index_pip),    # Index
            is_finger_extended(middle_tip, middle_pip),  # Middle
            is_finger_extended(ring_tip, ring_pip),      # Ring
            is_finger_extended(pinky_tip, pinky_pip)     # Pinky
        ]

        # Gesture recognition with normalized distances
        thumb_index_dist = distance(thumb_tip, index_tip) / palm_size
        index_middle_dist = distance(index_tip, middle_tip) / palm_size
        thumb_vertical_pos = (wrist.y - thumb_tip.y) / palm_size

        # Fist
        if not any(fingers):
            return "Fist"
        
        # Open Hand
        elif all(fingers):
            if distance(thumb_tip, pinky_tip) / palm_size > 0.7:
                return "Open Hand"
        
        # Peace Sign
        elif fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
            if 0.1 < index_middle_dist < 0.3:
                return "Peace"
        
        # Point
        elif fingers[1] and not any(fingers[0:1] + fingers[2:]):
            if index_tip.y < index_pip.y - 0.08:
                return "Point"
        
        # Thumbs Up/Down
        elif fingers[0] and not any(fingers[1:]):
            if thumb_vertical_pos > 0.15:
                return "Thumbs Up"
            elif thumb_vertical_pos < -0.15:
                return "Thumbs Down"
        
        # Rock On
        elif fingers[1] and fingers[4] and not fingers[0] and not fingers[2] and not fingers[3]:
            if distance(index_tip, pinky_tip) / palm_size > 0.6:
                return "Rock On"

        return "Unknown"

# ------------------------ Gender Classification ------------------------
class GenderClassifier:
    def __init__(self):
        try:
            self.gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.model_loaded = True
        except cv2.error as e:
            print(f"\nError loading gender classification model: {e}")
        
    def preprocess_face(self, face_img):
        """Preprocess face image for better classification"""
        # Convert to grayscale and apply histogram equalization
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Convert back to BGR for the model
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced

    def detect_faces(self, img):
        """Improved face detection with multiple scales"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),  # Minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces

    def predict_gender(self, face_img):
        """Predict gender from a face image with improved accuracy"""
        if not self.model_loaded:
            return None
            
        try:
            # Detect faces with improved detection
            faces = self.detect_faces(face_img)
            
            if len(faces) > 0:
                # Use the largest face
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                
                # Add padding to include more of the face
                padding = int(0.2 * w)  # 20% padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(face_img.shape[1] - x, w + 2*padding)
                h = min(face_img.shape[0] - y, h + 2*padding)
                
                face = face_img[y:y+h, x:x+w].copy()
                
                # Preprocess the face
                processed_face = self.preprocess_face(face)
                
                # Prepare input blob
                blob = cv2.dnn.blobFromImage(
                    processed_face, 
                    1.0, 
                    (227, 227), 
                    MODEL_MEAN_VALUES, 
                    swapRB=False
                )
                
                # Get prediction
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                
                # Get confidence and gender
                confidence = max(gender_preds[0])
                gender_idx = gender_preds[0].argmax()
                
                # Only return prediction if confidence is high enough
                if confidence > GENDER_CONFIDENCE_THRESHOLD:
                    return GENDER_LIST[gender_idx]
                
        except Exception:
            pass
        return None
    
    def detect_gender(self, frame):
        """Original method for standalone face detection and gender classification"""
        if not self.model_loaded:
            return
            
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Add padding
            padding = int(0.2 * w)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2*padding)
            h = min(frame.shape[0] - y, h + 2*padding)
            
            face_img = frame[y:y+h, x:x+w].copy()
            if face_img.size == 0:
                continue
                
            try:
                processed_face = self.preprocess_face(face_img)
                blob = cv2.dnn.blobFromImage(
                    processed_face, 
                    1.0, 
                    (227, 227), 
                    MODEL_MEAN_VALUES, 
                    swapRB=False
                )
                
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                confidence = max(gender_preds[0])
                
                if confidence > GENDER_CONFIDENCE_THRESHOLD:
                    gender = GENDER_LIST[gender_preds[0].argmax()]
                    
                    # Draw rectangle around face and label with gender
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, 
                              f"Gender: {gender} ({confidence:.2f})", 
                              (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.8, 
                              (0, 255, 255), 
                              2)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

# ------------------------ Main Application ------------------------
if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture(0)
        gender_classifier = GenderClassifier()
        yolo_detector = YOLOObjectDetector()
        hand_recognizer = HandGestureRecognizer()
        
        # Connect gender classifier to YOLO detector
        yolo_detector.set_gender_classifier(gender_classifier)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run all detections
            yolo_detector.detect_objects(frame)
            hand_recognizer.detect_hand_gestures(frame)
            
            cv2.imshow("Object Detection, Hand Gesture, and Gender Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
