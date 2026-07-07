import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(SCRIPT_DIR, "..", "calibration", "camera_params.yml")

# --- Load real calibrated intrinsics from Module 1 ---
fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
fs.release()
print("Loaded camera intrinsics:\n", K)

# --- ORB feature detector + matcher ---
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)

# --- State: accumulated camera pose ---
cur_R = np.eye(3)
cur_t = np.zeros((3, 1))

trajectory = [(0, 0)]

traj_canvas = np.zeros((600, 600, 3), dtype=np.uint8)
traj_scale = 50
traj_offset = (300, 300)

# --- Keyframe state: compare against a frame N steps back, not the immediately previous frame ---
FRAME_SKIP = 8   # try 5-15; higher = more parallax but fewer updates
key_kp = None
key_des = None
frame_count = 0

MIN_INLIER_RATIO = 0.5

print("Move your camera around slowly and steadily. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    display = frame.copy()
    frame_count += 1

    if key_des is not None and des is not None and len(kp) > 8 and len(key_kp) > 8 \
            and frame_count % FRAME_SKIP == 0:

        matches = bf.match(key_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) >= 8:
            pts_key = np.float32([key_kp[m.queryIdx].pt for m in matches])
            pts_cur = np.float32([kp[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(
                pts_cur, pts_key, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
            )

            if E is not None and E.shape == (3, 3):
                _, R, t, mask_pose = cv2.recoverPose(E, pts_cur, pts_key, K)
                inlier_count = int(mask_pose.sum()) if mask_pose is not None else 0
                inlier_ratio = inlier_count / len(matches)

                cv2.putText(display, f"Inliers: {inlier_count}/{len(matches)} ({inlier_ratio:.2f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if inlier_ratio >= MIN_INLIER_RATIO:
                    cur_t = cur_t + cur_R @ t
                    cur_R = R @ cur_R

                    x, z = cur_t[0, 0], cur_t[2, 0]
                    trajectory.append((x, z))

                    px = int(traj_offset[0] + x * traj_scale)
                    py = int(traj_offset[1] + z * traj_scale)
                    if 0 <= px < 600 and 0 <= py < 600:
                        cv2.circle(traj_canvas, (px, py), 2, (0, 255, 0), -1)
                else:
                    cv2.putText(display, "Pose REJECTED (low inlier ratio)", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        key_kp, key_des = kp, des

    elif key_des is None:
        key_kp, key_des = kp, des

    display = cv2.drawKeypoints(display, kp, None, color=(0, 255, 0))

    cv2.putText(display, f"Keypoints: {len(kp)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Camera Feed (ORB keypoints) - ESC to quit", display)
    cv2.imshow("Top-Down Trajectory (unscaled)", traj_canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal accumulated position (arbitrary scale): x={cur_t[0,0]:.2f}, z={cur_t[2,0]:.2f}")
print(f"Total trajectory points recorded: {len(trajectory)}")