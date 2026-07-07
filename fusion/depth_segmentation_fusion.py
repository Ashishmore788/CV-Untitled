import cv2
import torch
import numpy as np
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

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

cap = cv2.VideoCapture(0)
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # --- Fusion: average relative depth per detected class ---
    present_classes = np.unique(seg_pred_full)
    report_lines = []
    for c in present_classes:
        if c == 0:  # skip background
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

    cv2.imshow("Camera Feed + Fusion Labels", frame)
    cv2.imshow("Depth Map", depth_color)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)