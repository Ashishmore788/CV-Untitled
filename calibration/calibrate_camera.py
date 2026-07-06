import cv2
import numpy as np
import os
import glob

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "calib_images")
CHECKERBOARD = (9, 6)          # internal corners (must match capture script)
SQUARE_SIZE_MM = 16.93         # real-world size of one square, in mm (adjust if needed)
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "camera_params.yml")

# --- Prepare object points (real-world 3D points of checkerboard corners) ---
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
if len(images) == 0:
    print(f"No images found in {IMAGES_DIR}. Run the capture script first.")
    exit()

print(f"Found {len(images)} images. Processing...")

img_shape = None
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"  {os.path.basename(fname)}: corners found")
    else:
        print(f"  {os.path.basename(fname)}: corners NOT found (skipped)")

if len(objpoints) < 5:
    print(f"Only {len(objpoints)} valid images. Need at least 5-10 for a stable calibration.")
    exit()

print(f"\nRunning calibration using {len(objpoints)} valid images...")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print("\n=== Calibration Results ===")
print(f"RMS reprojection error: {ret:.4f} pixels  (lower is better; <0.5 is great, <1.0 is good)")
print("\nCamera matrix (K):")
print(K)
print("\nDistortion coefficients:")
print(dist.ravel())

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
print(f"\nfx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

fs = cv2.FileStorage(OUTPUT_FILE, cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", K)
fs.write("dist_coeff", dist)
fs.write("reprojection_error", ret)
fs.write("image_width", img_shape[0])
fs.write("image_height", img_shape[1])
fs.release()

print(f"\nSaved calibration to: {OUTPUT_FILE}")