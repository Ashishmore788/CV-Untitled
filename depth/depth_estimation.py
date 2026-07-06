import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image
import time

print("Loading Depth Anything V2 (small variant)... this may take a moment on first run.")

# device=-1 forces CPU, matching your Intel UHD / no-CUDA setup
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=-1
)

print("Model loaded. Starting webcam...")

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR (OpenCV) -> RGB (PIL/transformers expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Run depth estimation on this frame
    result = depth_pipe(pil_image)
    depth_map = np.array(result["depth"])  # relative depth, float values

    # Normalize depth map to 0-255 for visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # Apply a color map: warm = near, cool = far (INFERNO looks great for this)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    # Resize depth map to match original frame size (model may output different resolution)
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

    # Calculate FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show original and depth side-by-side
    combined = np.hstack((frame, depth_colored))
    cv2.imshow("RGB (left) | Depth (right) - press ESC to quit", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()