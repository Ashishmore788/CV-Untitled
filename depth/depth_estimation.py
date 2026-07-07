import cv2
import torch
import numpy as np

# --- Load MiDaS small model (fast, CPU-friendly) ---
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
print(f"Running on: {device}")

# --- Load matching transforms for this model type ---
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # --- Normalize for display (relative depth, not metric) ---
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_display = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Depth Map (relative)", depth_color)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)