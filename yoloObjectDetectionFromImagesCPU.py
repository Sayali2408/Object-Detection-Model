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
AGE_RANGES = ['(0-2)', '(3-7)', '(8-12)', '(13-17)', '(18-24)', '(25-32)', '(33-40)', '(41-50)', '(51-60)', '(61-70)', '(71-80)', '(81-90)', '(91-100)']

# ------------------------ Object Detection (YOLO) ------------------------
class YOLOObjectDetector:
    def __init__(self, weights_path="yolov3.weights", config_path="yolov3.cfg", names_path="coco.names"):
        self.yolo = cv2.dnn.readNet(weights_path, config_path)
        # Enable OpenCV DNN optimization
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
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
        # Resize frame for faster processing while maintaining aspect ratio
        scale = 0.5
        height, width = frame.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Create blob with optimized parameters
        blob = cv2.dnn.blobFromImage(resized_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)
        
        class_ids, confidences, boxes = [], [], []
        # Increased confidence threshold for better performance
        conf_threshold = 0.6
        nms_threshold = 0.3
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    # Scale back to original size
                    center_x = int(detection[0] * new_width / scale)
                    center_y = int(detection[1] * new_height / scale)
                    w = int(detection[2] * new_width / scale)
                    h = int(detection[3] * new_height / scale)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        # Batch process detections
        valid_detections = [(i, boxes[i], class_ids[i]) for i in indexes.flatten()]
        for i, box, class_id in valid_detections:
            x, y, w, h = box
            label = str(self.classes[class_id])
            
            # Process person detection only if confidence is very high
            if label == "person" and self.gender_classifier and self.gender_classifier.model_loaded and confidences[i] > 0.7:
                face_img = frame[max(0, y):min(y+h, height), max(0, x):min(x+w, width)].copy()
                if face_img.size > 0:
                    try:
                        gender = self.gender_classifier.predict_gender(face_img)
                        label = f"person ({gender})"
                    except Exception:
                        pass
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colorGreen, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, self.colorRed, 2)

# ------------------------ Hand Gesture Recognition (MediaPipe) ------------------------
class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hand_gestures(self, frame):
        # Resize frame for faster processing
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        height, width = frame.shape[:2]
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Scale landmarks back to original size
                scaled_landmarks = mp.solutions.hands.HandLandmark(
                    [landmark * (1/scale) for landmark in hand_landmarks.landmark]
                )
                self.mp_draw.draw_landmarks(frame, scaled_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.identify_gesture(scaled_landmarks)
                wrist = scaled_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
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

# ------------------------ Emotion Detection (DeepFace) ------------------------
class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_emotions(self, frame):
        try:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(60, 60)
            )
            
            for (x, y, w, h) in faces:
                try:
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Analyze emotion using DeepFace
                    result = DeepFace.analyze(
                        face_roi, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        silent=True
                    )
                    
                    # Get dominant emotion
                    emotion = result[0]['dominant_emotion']
                    
                    # Draw emotion label
                    cv2.putText(frame, 
                              f"Emotion: {emotion}", 
                              (x, y + h + 20),  # Position below face
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.8, 
                              (0, 0, 255),  # Red color
                              2)
                except Exception as e:
                    print(f"Error analyzing emotion: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in emotion detection: {e}")

# ------------------------ Age Detection ------------------------
class AgeDetector:
    def __init__(self):
        try:
            self.age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.model_loaded = True
            # Minimum confidence threshold for age predictions
            self.confidence_threshold = 0.6
        except cv2.error as e:
            print(f"\nError loading age detection model: {e}")
            self.model_loaded = False

    def assess_face_quality(self, face_img):
        """Assess the quality of the face image"""
        if face_img is None or face_img.size == 0:
            return False
            
        # Check minimum face size
        if face_img.shape[0] < 60 or face_img.shape[1] < 60:
            return False
            
        # Calculate image sharpness
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Threshold for blur detection
            return False
            
        # Check brightness
        brightness = np.mean(gray)
        if brightness < 40 or brightness > 250:  # Too dark or too bright
            return False
            
        return True

    def preprocess_face(self, face_img):
        """Enhanced preprocessing for face image"""
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            return None
            
        try:
            # Convert to grayscale and apply histogram equalization
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Create blob for the model
            blob = cv2.dnn.blobFromImage(
                enhanced, 
                1.0, 
                (227, 227),
                MODEL_MEAN_VALUES, 
                swapRB=False
            )
            return blob
        except Exception as e:
            print(f"Error in face preprocessing: {e}")
            return None

    def predict_age(self, face_img):
        """Enhanced age prediction using ensemble approach"""
        if face_img is None or face_img.size == 0:
            return None
            
        if not self.assess_face_quality(face_img):
            return None
            
        try:
            # Get prediction from Caffe model
            blob = self.preprocess_face(face_img)
            if blob is None:
                return None
                
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            confidence = np.max(age_preds[0])
            
            if confidence < self.confidence_threshold:
                return None
                
            age_idx = age_preds[0].argmax()
            caffe_age = AGE_RANGES[age_idx]
            
            # Get prediction from DeepFace
            try:
                deepface_result = DeepFace.analyze(
                    face_img, 
                    actions=['age'],
                    enforce_detection=False,
                    silent=True
                )
                deepface_age = deepface_result[0]['age']
                
                # Combine predictions
                caffe_age_range = caffe_age.strip('()').split('-')
                caffe_avg = (int(caffe_age_range[0]) + int(caffe_age_range[1])) / 2
                
                # Weighted average based on confidence
                final_age = int((caffe_avg + deepface_age) / 2)
                
                # Find the closest age range
                for age_range in AGE_RANGES:
                    range_nums = age_range.strip('()').split('-')
                    range_avg = (int(range_nums[0]) + int(range_nums[1])) / 2
                    if abs(range_avg - final_age) <= 5:
                        return age_range
                
                return caffe_age  # Fallback to Caffe prediction
                
            except Exception as e:
                print(f"DeepFace error: {e}")
                return caffe_age  # Fallback to Caffe prediction
                
        except Exception as e:
            print(f"Error predicting age: {str(e)}")
            return None

    def detect_age(self, frame):
        """Enhanced age detection in frame"""
        if not self.model_loaded:
            return
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhanced face detection parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # More precise scaling
                minNeighbors=6,    # Stricter detection
                minSize=(60, 60),  # Minimum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                try:
                    # Add padding to include more context
                    padding = int(0.1 * w)  # 10% padding
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    # Extract and preprocess face
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size == 0:
                        continue
                        
                    # Predict age
                    age_label = self.predict_age(face_roi)
                    
                    if age_label:
                        # Draw age label with better visibility
                        label_background = (0, 0, 0)
                        label_color = (255, 255, 255)
                        label_text = f"Age: {age_label}"
                        
                        # Get text size for background rectangle
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            2
                        )
                        
                        # Draw background rectangle
                        cv2.rectangle(
                            frame,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width + 10, y1),
                            label_background,
                            -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            frame,
                            label_text,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            label_color,
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
