import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(SCRIPT_DIR, "camera_params.yml")

# --- Load your real calibrated intrinsics ---
fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
fs.release()

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
print(f"Loaded real intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}\n")

# --- Step 1: Make up a 3D point in front of the camera (in mm, since we calibrated in mm) ---
X, Y, Z = 50.0, 30.0, 500.0
print(f"Original 3D point (mm): X={X}, Y={Y}, Z={Z}")

# --- Step 2: Project it to a pixel using the pinhole camera model ---
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
print(f"Projected to pixel: u={u:.2f}, v={v:.2f}")

# --- Step 3: Deproject that pixel back to 3D, using the SAME depth Z ---
X_recovered = (u - cx) * Z / fx
Y_recovered = (v - cy) * Z / fy
print(f"Recovered 3D point (mm): X={X_recovered:.2f}, Y={Y_recovered:.2f}, Z={Z}")

# --- Step 4: Confirm it matches ---
error = np.sqrt((X - X_recovered)**2 + (Y - Y_recovered)**2)
print(f"\nRound-trip error: {error:.6f} mm  (should be ~0, confirming the math is consistent)")