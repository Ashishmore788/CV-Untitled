import cv2
import os

# Anchor SAVE_DIR to this script's location, not the current working directory,
# so it works correctly whether you run it from project root or from this folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "calib_images")
CHECKERBOARD = (9, 6)  # internal corners, adjust to your printed board

os.makedirs(SAVE_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)

count = 0
print("Press 'c' to capture a frame, ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()
    if found:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
        cv2.putText(display, "Checkerboard detected - press C to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No checkerboard detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(display, f"Captured: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Calibration Capture", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key in (ord('c'), ord('C')) and found:  # 'c' key to capture
        path = os.path.join(SAVE_DIR, f"calib_{count:02d}.jpg")
        cv2.imwrite(path, frame)
        print(f"Saved {path}")
        count += 1

cap.release()
cv2.destroyAllWindows()
print(f"Done. Captured {count} images in '{SAVE_DIR}/'")