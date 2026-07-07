import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
model = lraspp_mobilenet_v3_large(weights=weights)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running on: {device}")

preprocess = weights.transforms()
categories = weights.meta["categories"]
print(f"Loaded {len(categories)} classes: {categories}")

# --- Random color per class, fixed seed for consistency across frames ---
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(categories), 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Downscale before inference for speed ---
    small_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    pred = output.argmax(0).byte().cpu().numpy()

    # Resize mask back to ORIGINAL frame size (not the downscaled size)
    pred_resized = cv2.resize(pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    # --- Build color overlay ---
    color_mask = colors[pred_resized]
    overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

    # --- Show which classes are present this frame ---
    present_classes = np.unique(pred_resized)
    labels_text = ", ".join(categories[c] for c in present_classes if c != 0)
    cv2.putText(overlay, labels_text[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Semantic Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # flush GUI events so the window actually closes on Windows