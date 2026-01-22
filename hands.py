import cv2
import numpy as np
import os
import time
from collections import deque
import json

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# Constants
KERNEL_SIZE = 5
DILATE_ITERATIONS = 2
BLUR_SIZE = 5
FINGER_ANGLE_THRESHOLD = 90  # degrees
MIN_CONTOUR_AREA = 5000
MIN_DEFECT_DISTANCE = 10000  # minimum depth of defect
SMOOTH_WINDOW = 5  # frames for smoothing finger count
EXTENT_FIST_THRESHOLD = 0.7  # contour extent threshold to detect fist vs single finger
CONTOUR_APPROX_EPSILON = 0.02  # epsilon for contour approximation
SETTINGS_FILE = "hsv_settings.json"

# Gesture labels
GESTURES = {
    0: "Fist",
    1: "One",
    2: "Peace/Two",
    3: "Three",
    4: "Four",
    5: "Five/Open Hand"
}

# HSV trackbar callback (does nothing, just for trackbar)
def nothing(x):
    pass

def load_hsv_settings():
    """Load HSV settings from file"""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def save_hsv_settings(h_l, s_l, v_l, h_u, s_u, v_u):
    """Save HSV settings to file"""
    settings = {
        'h_lower': h_l, 's_lower': s_l, 'v_lower': v_l,
        'h_upper': h_u, 's_upper': s_u, 'v_upper': v_u
    }
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)
    print("HSV settings saved!")

# Load saved settings or use defaults
saved_settings = load_hsv_settings()
if saved_settings:
    h_l, s_l, v_l = saved_settings['h_lower'], saved_settings['s_lower'], saved_settings['v_lower']
    h_u, s_u, v_u = saved_settings['h_upper'], saved_settings['s_upper'], saved_settings['v_upper']
    print("Loaded saved HSV settings")
else:
    h_l, s_l, v_l = 0, 20, 70
    h_u, s_u, v_u = 20, 255, 255

# Create window and trackbars for HSV tuning
cv2.namedWindow("HSV Tuning")
cv2.createTrackbar("H Lower", "HSV Tuning", h_l, 179, nothing)
cv2.createTrackbar("S Lower", "HSV Tuning", s_l, 255, nothing)
cv2.createTrackbar("V Lower", "HSV Tuning", v_l, 255, nothing)
cv2.createTrackbar("H Upper", "HSV Tuning", h_u, 179, nothing)
cv2.createTrackbar("S Upper", "HSV Tuning", s_u, 255, nothing)
cv2.createTrackbar("V Upper", "HSV Tuning", v_u, 255, nothing)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# Performance tracking
fps_buffer = deque(maxlen=30)
finger_count_buffer = deque(maxlen=SMOOTH_WINDOW)
prev_time = time.time()

print("Controls:")
print("  ESC - Exit")
print("  'r' - Reset HSV to defaults")
print("  's' - Save current HSV settings")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    fps_buffer.append(fps)
    avg_fps = sum(fps_buffer) / len(fps_buffer)

    frame = cv2.flip(frame, 1)  # mirror image
    frame_display = frame.copy()  # copy for drawing

    # ---------------- PREPROCESS ----------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    h_lower = cv2.getTrackbarPos("H Lower", "HSV Tuning")
    s_lower = cv2.getTrackbarPos("S Lower", "HSV Tuning")
    v_lower = cv2.getTrackbarPos("V Lower", "HSV Tuning")
    h_upper = cv2.getTrackbarPos("H Upper", "HSV Tuning")
    s_upper = cv2.getTrackbarPos("S Upper", "HSV Tuning")
    v_upper = cv2.getTrackbarPos("V Upper", "HSV Tuning")
    
    lower_skin = np.array([h_lower, s_lower, v_lower])
    upper_skin = np.array([h_upper, s_upper, v_upper])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=DILATE_ITERATIONS)
    mask = cv2.GaussianBlur(mask, (BLUR_SIZE, BLUR_SIZE), 0)

    # ---------------- CONTOURS ----------------
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    finger_count = 0
    contour_area = 0
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(cnt)
        
        # Filter out small contours (noise)
        if contour_area >= MIN_CONTOUR_AREA:
            # Approximate contour to reduce points
            epsilon = CONTOUR_APPROX_EPSILON * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            hull = cv2.convexHull(cnt)
            cv2.drawContours(frame_display, [hull], -1, (0, 255, 0), 2)

            hull_indices = cv2.convexHull(cnt, returnPoints=False)

            if hull_indices is not None and len(hull_indices) > 3:
                defects = cv2.convexityDefects(cnt, hull_indices)

                # default values
                valid_defects = 0
                extent = 0.0

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]

                        # Skip if defect is too shallow
                        if d < MIN_DEFECT_DISTANCE:
                            continue

                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])

                        # Triangle sides
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(end) - np.array(far))

                        # Cosine rule (with safety check)
                        denom = 2 * b * c
                        if denom == 0:
                            continue
                        
                        cos_angle = (b**2 + c**2 - a**2) / denom
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # prevent domain error
                        angle = np.arccos(cos_angle)
                        angle_deg = np.degrees(angle)

                        # Count valid defect if angle < threshold
                        if angle_deg <= FINGER_ANGLE_THRESHOLD:
                            valid_defects += 1
                            cv2.circle(frame_display, far, 5, (0, 0, 255), -1)

                # If no valid defects, distinguish between a closed fist and a single extended finger
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h if w * h > 0 else 1
                extent = contour_area / rect_area
                if valid_defects == 0:
                    if extent > EXTENT_FIST_THRESHOLD:
                        finger_count = 0
                    else:
                        finger_count = 1
                else:
                    # number of fingers = defects + 1
                    finger_count = valid_defects + 1

                # Smooth finger count using buffer (append once)
                finger_count_buffer.append(finger_count)
                finger_count = int(np.median(finger_count_buffer)) if len(finger_count_buffer) > 0 else finger_count

                # Debug overlay for tuning
                cv2.putText(
                    frame_display,
                    f"defects:{valid_defects} extent:{extent:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                
                # Get gesture label
                gesture = GESTURES.get(finger_count, f"{finger_count} fingers")
                
                # Display finger count and gesture
                cv2.putText(
                    frame_display,
                    f"Fingers: {finger_count} - {gesture}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

    # ---------------- DISPLAY ----------------
    # Display FPS
    cv2.putText(
        frame_display,
        f"FPS: {avg_fps:.1f}",
        (10, frame_display.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    
    # Display contour area
    if contour_area > 0:
        cv2.putText(
            frame_display,
            f"Area: {int(contour_area)}",
            (10, frame_display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
    
    cv2.imshow("Finger Counting", frame_display)
    cv2.imshow("Mask", mask)
    cv2.imshow("HSV Tuning", np.zeros((1, 400, 3), dtype=np.uint8))  # placeholder

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('r'):  # Reset trackbars to defaults
        cv2.setTrackbarPos("H Lower", "HSV Tuning", 0)
        cv2.setTrackbarPos("S Lower", "HSV Tuning", 20)
        cv2.setTrackbarPos("V Lower", "HSV Tuning", 70)
        cv2.setTrackbarPos("H Upper", "HSV Tuning", 20)
        cv2.setTrackbarPos("S Upper", "HSV Tuning", 255)
        cv2.setTrackbarPos("V Upper", "HSV Tuning", 255)
        print("HSV settings reset to defaults")
    elif key == ord('s'):  # Save current settings
        h_l = cv2.getTrackbarPos("H Lower", "HSV Tuning")
        s_l = cv2.getTrackbarPos("S Lower", "HSV Tuning")
        v_l = cv2.getTrackbarPos("V Lower", "HSV Tuning")
        h_u = cv2.getTrackbarPos("H Upper", "HSV Tuning")
        s_u = cv2.getTrackbarPos("S Upper", "HSV Tuning")
        v_u = cv2.getTrackbarPos("V Upper", "HSV Tuning")
        save_hsv_settings(h_l, s_l, v_l, h_u, s_u, v_u)

cap.release()
cv2.destroyAllWindows()
