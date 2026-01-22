import cv2
import numpy as np


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Object Counting App Started")
    print("Keys: q=quit, r=reset, b=capture background, +=increase area, -=decrease area")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    min_area = 500
    thresh_val = 25
    blur_ksize = 21

    # Controls window with trackbars
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Min Area", "Controls", min_area, 5000, nothing)
    cv2.createTrackbar("Threshold", "Controls", thresh_val, 255, nothing)
    cv2.createTrackbar("Blur", "Controls", blur_ksize, 51, nothing)

    bg_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        display_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # read trackbar values
        min_area = max(1, cv2.getTrackbarPos("Min Area", "Controls"))
        thresh_val = cv2.getTrackbarPos("Threshold", "Controls")
        blur_ksize = cv2.getTrackbarPos("Blur", "Controls")
        if blur_ksize % 2 == 0:
            blur_ksize = max(1, blur_ksize - 1)

        blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        if bg_frame is not None:
            # Use simple frame differencing against captured background
            frameDelta = cv2.absdiff(bg_frame, blurred)
            _, thresh = cv2.threshold(frameDelta, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            # Fall back to MOG2 background subtractor
            fg_mask = bg_subtractor.apply(blurred)
            # Remove shadows (if any) by thresholding low values
            _, thresh = cv2.threshold(fg_mask, max(10, thresh_val), 255, cv2.THRESH_BINARY)

        # Morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                object_count += 1
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(display_frame, [contour], -1, (0, 255, 255), 1)
                M = cv2.moments(contour)
                if M.get("m00", 0) != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_frame, (cx, cy), 5, (255, 0, 0), -1)

        # Overlay info
        cv2.rectangle(display_frame, (10, 10), (320, 90), (0, 0, 0), -1)
        cv2.putText(display_frame, f"Objects Detected: {object_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        mode_text = "Mode: FrameDiff" if bg_frame is not None else "Mode: MOG2"
        cv2.putText(display_frame, f"{mode_text}  MinArea: {min_area}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Object Counter", display_frame)
        cv2.imshow("Threshold Mask", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
            bg_frame = None
            print("Background models reset")
        elif key == ord('b'):
            # Capture background (ask user to show empty scene)
            bg_frame = blurred.copy()
            print("Background captured for frame differencing")
        elif key == ord('+') or key == ord('='):
            min_area += 100
            cv2.setTrackbarPos("Min Area", "Controls", min_area)
            print(f"Min area: {min_area}")
        elif key == ord('-') or key == ord('_'):
            min_area = max(1, min_area - 100)
            cv2.setTrackbarPos("Min Area", "Controls", min_area)
            print(f"Min area: {min_area}")

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")


if __name__ == "__main__":
    main()