
# ---------------- FIX WINDOWS CAMERA BACKEND ----------------
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np

# --------- PIXEL TO MM SCALE ---------
try:
    pixels_per_mm = float(input("Enter the number of pixels per 1 mm (e.g., 10): "))
    if pixels_per_mm <= 0:
        raise ValueError
except Exception:
    print("Invalid input. Using default: 10 pixels per mm.")
    pixels_per_mm = 10.0

def px_to_mm(px):
    return px / pixels_per_mm

def circumference_mm(radius_px):
    radius_mm = px_to_mm(radius_px)
    return 2 * np.pi * radius_mm

# ---------------- ZOOM PARAMETERS ----------------
zoom_factor = 1.0
ZOOM_STEP = 0.1
MAX_ZOOM = 4.0
MIN_ZOOM = 1.0

def apply_zoom(frame, center_x, center_y, zoom):
    if zoom <= 1.0:
        return frame

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]
    cam_cx, cam_cy = w0 // 2, h0 // 2

    # -------- APPLY ZOOM --------
    frame = apply_zoom(frame, cam_cx, cam_cy, zoom_factor)

    # -------- CAMERA CENTER --------
    h, w = frame.shape[:2]
    cam_cx, cam_cy = w // 2, h // 2

    cv2.circle(frame, (cam_cx, cam_cy), 5, (0, 255, 255), -1)

    # -------- PREPROCESS --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # -------- OUTER CIRCLE --------
    outer_circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=120,
        param2=40,
        minRadius=30,
        maxRadius=150
    )

    if outer_circles is not None:
        ox, oy, outer_r = np.uint16(np.around(outer_circles[0][0]))

        cv2.circle(frame, (ox, oy), outer_r, (255, 0, 0), 2)
        cv2.circle(frame, (ox, oy), 3, (255, 0, 0), -1)

        # -------- MASK FOR INNER --------
        mask = np.zeros_like(gray)
        cv2.circle(mask, (ox, oy), outer_r - 5, 255, -1)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        masked_blur = cv2.GaussianBlur(masked, (9, 9), 0)

        # -------- INNER CIRCLE --------
        inner_circles = cv2.HoughCircles(
            masked_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=25,
            minRadius=10,
            maxRadius=outer_r - 15
        )

        if inner_circles is not None:
            ix, iy, inner_r = np.uint16(np.around(inner_circles[0][0]))

            cv2.circle(frame, (ix, iy), inner_r, (0, 255, 0), 2)
            cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1)

            # -------- OFFSET --------
            offset_x = ox - cam_cx
            offset_y = oy - cam_cy
            offset_dist = int((offset_x**2 + offset_y**2) ** 0.5)

            cv2.line(frame, (cam_cx, cam_cy), (ox, oy), (0, 255, 255), 2)

            # -------- TEXT --------

            # --- Display px and mm values ---
            outer_r_mm = px_to_mm(outer_r)
            inner_r_mm = px_to_mm(inner_r)
            outer_circ_mm = circumference_mm(outer_r)
            inner_circ_mm = circumference_mm(inner_r)

            cv2.putText(frame, f"Outer R: {outer_r} px / {outer_r_mm:.2f} mm",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Outer Circ: {outer_circ_mm:.2f} mm",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, f"Inner R: {inner_r} px / {inner_r_mm:.2f} mm",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inner Circ: {inner_circ_mm:.2f} mm",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Offset: {offset_dist} px",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # -------- ZOOM DISPLAY --------
    cv2.putText(frame, f"Zoom: {zoom_factor:.1f}x",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Live Shim Detection", frame)

    # -------- KEYBOARD --------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        zoom_factor = min(MAX_ZOOM, zoom_factor + ZOOM_STEP)
    elif key == ord('-'):
        zoom_factor = max(MIN_ZOOM, zoom_factor - ZOOM_STEP)
    elif key == ord('r'):
        zoom_factor = 1.0
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
