import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import os
import sys

# constants of the file
GENDER_MODEL = 'gender_net.caffemodel'
GENDER_PROTO = 'gender_deploy.prototxt'
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_CONFIDENCE_THRESHOLD = 0.8
AGE_MODEL = 'age_net.caffemodel'
AGE_PROTO = 'age_deploy.prototxt'
AGE_RANGES = [(0, 2), (3, 7), (8, 12), (13, 17), (18, 24), (25, 32), (33, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]

# ------------------------ Object Detection (YOLO) ------------------------
class YOLOObjectDetector:
    def __init__(self, weights_path="yolov3.weights", config_path="yolov3.cfg", names_path="coco.names"):
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
            # Add prediction cache
            self.prediction_cache = {}
            self.cache_ttl = 30  # Cache predictions for 30 frames
            self.frame_count = 0
        except cv2.error as e:
            print(f"\nError loading gender classification model: {e}")
            self.model_loaded = False
    
    def preprocess_face(self, face_img):
        """Enhanced preprocessing pipeline for better classification"""
        if face_img.size == 0:
            return None
            
        # Resize to optimal size for gender detection
        face_img = cv2.resize(face_img, (227, 227))
        
        # Convert to grayscale and apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Convert back to BGR for the model
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced

    def detect_faces(self, img):
        """Optimized face detection with scale factor tuning"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Smaller scale factor for better detection
            minNeighbors=5,   # Increased for more reliable detection
            minSize=(30, 30)  # Minimum face size
        )
        return faces

    def predict_gender(self, face_img, face_encoding=None):
        """Predict gender with caching and confidence threshold"""
        # Check cache first
        if face_encoding is not None:
            cache_key = str(face_encoding)
            if cache_key in self.prediction_cache:
                cached_pred = self.prediction_cache[cache_key]
                if cached_pred['frame_count'] > self.frame_count - self.cache_ttl:
                    return cached_pred['gender']

        # Preprocess face
        processed_face = self.preprocess_face(face_img)
        if processed_face is None:
            return "Unknown"

        # Create blob and get prediction
        blob = cv2.dnn.blobFromImage(processed_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        
        # Get prediction with confidence
        gender_idx = gender_preds[0].argmax()
        confidence = gender_preds[0][gender_idx]
        
        # Only return prediction if confidence is high enough
        if confidence > GENDER_CONFIDENCE_THRESHOLD:
            gender = GENDER_LIST[gender_idx]
            
            # Cache prediction
            if face_encoding is not None:
                self.prediction_cache[cache_key] = {
                    'gender': gender,
                    'frame_count': self.frame_count
                }
            
            return gender
        return "Unknown"

    def process_frame(self, frame):
        """Process multiple faces in a frame efficiently"""
        if not self.model_loaded:
            return frame

        self.frame_count += 1
        faces = self.detect_faces(frame)
        
        # Process faces in batch
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w].copy()
            
            # Get face encoding for caching
            try:
                face_encoding = cv2.resize(face_img, (64, 64)).flatten()
            except Exception:
                face_encoding = None
            
            gender = self.predict_gender(face_img, face_encoding)
            
            # Draw results
            color = (0, 255, 0) if gender != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

# ------------------------ Emotion Detection (DeepFace) ------------------------
class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Initialize DeepFace with optimized settings
        self.emotion_threshold = 0.4
        # Cache to store recent predictions
        self.prediction_cache = {}
        self.cache_ttl = 15  # Cache predictions for 15 frames
        self.frame_count = 0

    def detect_emotions(self, frame):
        """Detect emotions in frame with error handling and caching"""
        if frame is None or frame.size == 0:
            return frame

        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            try:
                # Extract face ROI
                face_img = frame[y:y+h, x:x+w].copy()
                
                # Skip if face is too small
                if face_img.shape[0] < 64 or face_img.shape[1] < 64:
                    continue

                # Generate cache key from face position and size
                cache_key = f"{x}_{y}_{w}_{h}"
                
                # Check cache
                if cache_key in self.prediction_cache:
                    cached_pred = self.prediction_cache[cache_key]
                    if cached_pred['frame_count'] > self.frame_count - self.cache_ttl:
                        emotion = cached_pred['emotion']
                        confidence = cached_pred['confidence']
                    else:
                        del self.prediction_cache[cache_key]
                        emotion, confidence = self._predict_emotion(face_img)
                else:
                    emotion, confidence = self._predict_emotion(face_img)

                # Cache prediction
                if confidence > self.emotion_threshold:
                    self.prediction_cache[cache_key] = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'frame_count': self.frame_count
                    }

                    # Draw emotion label above the face
                    label = f"Emotion: {emotion} ({confidence:.2f})"
                    cv2.putText(frame, label,
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.6, (255, 255, 0), 2)  # Yellow color

            except Exception as e:
                print(f"Error processing face for emotion: {str(e)}")
                continue

        return frame

    def _predict_emotion(self, face_img):
        """Predict emotion with enhanced error handling"""
        try:
            # Ensure minimum size for DeepFace
            face_img = cv2.resize(face_img, (48, 48))
            
            # Use DeepFace for emotion prediction with optimized settings
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]  # Get first result if multiple faces detected
            
            if result and 'emotion' in result:
                emotions = result['emotion']
                # Get emotion with highest confidence
                emotion = max(emotions.items(), key=lambda x: x[1])
                return emotion[0].capitalize(), emotion[1] / 100.0
                
        except Exception as e:
            print(f"DeepFace emotion prediction error: {str(e)}")
        
        return "Unknown", 0.0

# ------------------------ Age Detection ------------------------
class AgeDetector:
    def __init__(self):
        try:
            self.age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.model_loaded = True
            
            # More granular age ranges and their midpoints
            self.age_ranges = [
                (0, 2),    # Babies
                (3, 7),    # Young children
                (8, 12),   # Children
                (13, 17),  # Teenagers
                (18, 24),  # Young adults
                (25, 32),  # Adults
                (33, 40),  # Middle-aged
                (41, 50),  # Middle-aged
                (51, 60),  # Mature adults
                (61, 70),  # Seniors
                (71, 80),  # Elderly
                (81, 90),  # Elderly
                (91, 100)  # Elderly
            ]
            
            # Initialize moving average for age predictions
            self.age_history = {}
            self.history_size = 5
            self.min_confidence = 0.6
            
        except cv2.error as e:
            print(f"\nError loading age detection model: {e}")
            self.model_loaded = False

    def preprocess_face(self, face_img):
        """Enhanced preprocessing for better age detection"""
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            return None
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Resize to model input size
            enhanced = cv2.resize(enhanced, (227, 227))
            
            # Create blob with enhanced preprocessing
            blob = cv2.dnn.blobFromImage(
                enhanced, 
                1.0, 
                (227, 227),
                MODEL_MEAN_VALUES,
                swapRB=False
            )
            return blob
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def smooth_age_prediction(self, face_id, age, confidence):
        """Apply moving average to age predictions"""
        if confidence < self.min_confidence:
            return age, confidence
            
        if face_id not in self.age_history:
            self.age_history[face_id] = []
            
        self.age_history[face_id].append((age, confidence))
        
        # Keep only recent predictions
        if len(self.age_history[face_id]) > self.history_size:
            self.age_history[face_id].pop(0)
            
        # Calculate weighted average based on confidence
        total_weight = 0
        weighted_age = 0
        
        for hist_age, hist_conf in self.age_history[face_id]:
            weighted_age += hist_age * hist_conf
            total_weight += hist_conf
            
        smoothed_age = int(round(weighted_age / total_weight))
        avg_confidence = total_weight / len(self.age_history[face_id])
        
        return smoothed_age, avg_confidence

    def get_age_range(self, age):
        """Get appropriate age range text"""
        for i, (min_age, max_age) in enumerate(self.age_ranges):
            if min_age <= age <= max_age:
                return f"{min_age}-{max_age}"
        return "Unknown"

    def predict_age(self, face_img):
        """Predict age with improved accuracy"""
        if face_img is None or face_img.size == 0:
            return None, 0
            
        try:
            blob = self.preprocess_face(face_img)
            if blob is None:
                return None, 0
                
            self.age_net.setInput(blob)
            pred = self.age_net.forward()
            
            # Get prediction and confidence
            age_idx = pred[0].argmax()
            confidence = float(pred[0][age_idx])
            
            # Get age range
            min_age, max_age = self.age_ranges[age_idx]
            predicted_age = (min_age + max_age) // 2
            
            return predicted_age, confidence
            
        except Exception as e:
            print(f"\nError predicting age: {str(e)}")
            return None, 0

    def detect_age(self, frame):
        """Detect age with improved visualization"""
        if not self.model_loaded:
            return
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                try:
                    # Extract and preprocess face
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        continue
                        
                    # Generate face ID for tracking
                    face_id = f"{x}_{y}_{w}_{h}"
                    
                    # Predict age with confidence
                    age, confidence = self.predict_age(face_roi)
                    
                    if age is not None:
                        # Apply smoothing
                        smoothed_age, avg_confidence = self.smooth_age_prediction(face_id, age, confidence)
                        
                        # Get age range
                        age_range = self.get_age_range(smoothed_age)
                        
                        # Draw age label below the face
                        label = f'Age: {smoothed_age} ({age_range}) ({avg_confidence:.2f})'
                        cv2.putText(
                            frame,
                            label,
                            (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 255),  # Magenta color
                            2
                        )
                except Exception as e:
                    print(f"Error processing face for age: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in age detection: {e}")

# Add error handling for missing model files
def check_required_files():
    required_files = {
        'YOLO': ['yolov3.weights', 'yolov3.cfg', 'coco.names'],
        'Gender': ['gender_net.caffemodel', 'gender_deploy.prototxt'],
        'Age': ['age_net.caffemodel', 'age_deploy.prototxt']
    }
    
    missing_files = []
    for category, files in required_files.items():
        for file in files:
            if not os.path.exists(file):
                missing_files.append(f"{category}: {file}")
    
    if missing_files:
        print("\nMissing required files:")
        for file in missing_files:
            print(f"- {file}")
        print("\nPlease ensure all model files are in the project directory.")
        return False
    return True

# ------------------------ Main Application ------------------------
if __name__ == "__main__":
    try:
        # Check required files before starting
        if not check_required_files():
            sys.exit(1)
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open video capture device")
            
        # Initialize detectors with better error handling
        gender_classifier = GenderClassifier()
        yolo_detector = YOLOObjectDetector()
        hand_recognizer = HandGestureRecognizer()
        emotion_detector = EmotionDetector()
        age_detector = AgeDetector()
        
        # Connect gender classifier to YOLO detector
        yolo_detector.set_gender_classifier(gender_classifier)
        
        print("\nStarting detection system...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Run all detections with try-except blocks
            try:
                yolo_detector.detect_objects(frame)
                hand_recognizer.detect_hand_gestures(frame)
                emotion_detector.detect_emotions(frame)
                age_detector.detect_age(frame)
                
                # Add FPS counter
                cv2.putText(frame, 
                           f"Press 'q' to quit", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, 
                           (0, 255, 255), 
                           2)
                
                cv2.imshow("Multi-Modal Detection System", frame)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping detection system...")
                break
    
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
