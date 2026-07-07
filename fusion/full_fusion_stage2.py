import cv2
import torch
import numpy as np
import os
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(SCRIPT_DIR, "..", "calibration", "camera_params.yml")

# --- Load calibrated intrinsics (for VO) ---
fs = cv2.FileStorage(CALIB_FILE, cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
fs.release()

# --- Segmentation model ---
seg_weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
seg_model = lraspp_mobilenet_v3_large(weights=seg_weights)
seg_model.eval()
seg_preprocess = seg_weights.transforms()
categories = seg_weights.meta["categories"]

# --- Depth model ---
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
depth_transform = midas_transforms.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seg_model.to(device)
midas.to(device)
print(f"Running on: {device}")

# --- VO setup (ORB + keyframe skipping, same as Module 3) ---
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cur_R = np.eye(3)
cur_t = np.zeros((3, 1))

FRAME_SKIP = 8
MIN_INLIER_RATIO = 0.5
key_kp, key_des = None, None
frame_count = 0

cap = cv2.VideoCapture(0)
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    small_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # --- Segmentation ---
    seg_input = seg_preprocess(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_out = seg_model(seg_input)["out"][0]
    seg_pred = seg_out.argmax(0).byte().cpu().numpy()
    seg_pred_full = cv2.resize(seg_pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # --- Depth ---
    depth_input = depth_transform(rgb).to(device)
    with torch.no_grad():
        depth_pred = midas(depth_input)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = depth_pred.cpu().numpy()

    # --- VO (keyframe-based, same logic as Module 3) ---
    pose_updated = False
    if key_des is not None and des is not None and len(kp) > 8 and len(key_kp) > 8 \
            and frame_count % FRAME_SKIP == 0:
        matches = bf.match(key_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) >= 8:
            pts_key = np.float32([key_kp[m.queryIdx].pt for m in matches])
            pts_cur = np.float32([kp[m.trainIdx].pt for m in matches])
            E, mask = cv2.findEssentialMat(pts_cur, pts_key, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and E.shape == (3, 3):
                _, R, t, mask_pose = cv2.recoverPose(E, pts_cur, pts_key, K)
                inlier_count = int(mask_pose.sum()) if mask_pose is not None else 0
                inlier_ratio = inlier_count / len(matches)
                if inlier_ratio >= MIN_INLIER_RATIO:
                    cur_t = cur_t + cur_R @ t
                    cur_R = R @ cur_R
                    pose_updated = True
        key_kp, key_des = kp, des
    elif key_des is None:
        key_kp, key_des = kp, des

    # --- Fusion: per-class relative depth ---
    present_classes = np.unique(seg_pred_full)
    report_lines = [f"Pose (x,z): {cur_t[0,0]:.2f}, {cur_t[2,0]:.2f}" + (" [updated]" if pose_updated else "")]
    for c in present_classes:
        if c == 0:
            continue
        mask = seg_pred_full == c
        avg_depth = depth_map[mask].mean()
        report_lines.append(f"{categories[c]}: {avg_depth:.1f}")

    # --- Display ---
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    y = 30
    for line in report_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25

    cv2.imshow("Camera Feed + Fusion + VO", frame)
    cv2.imshow("Depth Map", depth_color)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)