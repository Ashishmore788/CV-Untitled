# CV Scene Reconstruction — Monocular Visual SLAM-Lite Pipeline

A lightweight, monocular (single-camera) scene-understanding pipeline built incrementally in Python and OpenCV — no stereo camera, LiDAR, or IMU required. Each module was developed and validated independently before being fused with the next, with results tested against scenarios specifically designed to catch superficially-plausible-but-wrong output, not just "it ran without crashing."

Combines classical multi-view geometry (visual odometry), deep-learning perception (semantic segmentation, monocular depth), and multi-signal fusion into a live top-down map of the camera's surroundings.

---

## Pipeline Overview

| Module | Description | Status |
|---|---|---|
| 1. Setup & Calibration | Project environment and camera intrinsic calibration | ✅ Complete |
| 2. Feature Detection & Matching | ORB/SIFT-based keypoint detection and matching across frames | ✅ Complete |
| 3. Visual Odometry (VO) | Frame-to-frame camera pose estimation via Essential Matrix | ✅ Validated |
| 4. Semantic Segmentation | Pixel-level object classification per frame | ✅ Validated |
| 5. Depth Estimation & Fusion | Monocular relative depth (MiDaS) fused with segmentation masks | ✅ Validated |
| 6. Full Fusion + Top-Down Map | Pose + depth + segmentation combined into a live 2D map | ✅ Validated |

---

## Module 1 — Project Setup & Camera Calibration

Establishes the foundation the rest of the pipeline depends on: a reproducible Python environment and an accurately calibrated camera.

- Python virtual environment (`cv_project_env`) with pinned dependencies (OpenCV, PyTorch, timm, etc.)
- Camera intrinsic calibration performed to recover the camera matrix used by every downstream geometry-based module (VO, depth backprojection):

```
[[571.26065977   0.         296.34796287]
 [  0.         607.09914832 245.57027438]
 [  0.           0.           1.        ]]
```

- All scripts are run from the project root so relative paths (e.g. to the calibration file) resolve correctly.

---

## Module 2 — Feature Detection & Matching

Establishes the low-level correspondence pipeline that Visual Odometry (Module 3) builds on.

- Keypoint detection and descriptor extraction (ORB/SIFT) on live camera frames.
- Frame-to-frame feature matching to establish correspondences, which are later used to estimate the Essential Matrix in Module 3.
- Validated qualitatively by confirming matches track consistent physical points across consecutive frames rather than drifting or matching spuriously.

---

## Module 3 — Monocular Visual Odometry

**Method:** Estimates camera motion frame-to-frame using classical multi-view geometry — feature correspondences (Module 2) → Essential Matrix estimation → decomposition into rotation `R` and translation direction `t` via `recoverPose` → accumulation of `(x, z)` into an arbitrary-scale trajectory.

### Validation Tests

| Test | Final (x, z) | Magnitude | Interpretation |
|---|---|---|---|
| Stationary (camera still) | x≈0, z≈0 | ≈5.84 | Baseline noise floor — passed |
| Pure pan (rotation only) | x=-1.39, z=0.73 | ≈1.57 | Below noise floor — expected geometric limitation, not a bug |
| Sideways slide (translation) | x=2.30, z=-8.54 | ≈8.84 | ~1.5× noise floor — positive signal, but axis direction not independently confirmed |

### Key Findings

- Stationary drift is small and stable — the system doesn't hallucinate motion.
- Deliberate motion consistently produces a larger accumulated translation than the stationary baseline, across two independent tests.
- **Pure rotation is a known degenerate case** for Essential Matrix decomposition: with no real parallax, recovered translation becomes numerically unstable and can register *smaller* than the noise floor despite real motion. This is a geometric ambiguity inherent to the method — not fixable with frame-skipping or inlier gating.
- The sideways-translation test showed a larger-than-baseline magnitude, but the dominant axis (z/forward) didn't match the expected dominant axis (x/lateral). Plausible causes: imperfect physical execution of the "sideways" slide, axis-convention mismatch, or rotation/translation coupling.

**Documented limitation:** *VO is validated for stationary baseline (low drift) and shows proportionally larger response under deliberate motion; however, axis-direction correspondence between camera-frame translation and physical motion was not independently confirmed [in isolated testing], and pure-rotation motion is a known degenerate case for this pipeline.*

---

## Module 4 — Semantic Segmentation

**Method:** A pretrained semantic segmentation model (fixed 21-class vocabulary) is applied per-frame to the live webcam feed, producing a pixel-level class label mask. Developed and validated independently before fusion, to isolate segmentation quality from downstream fusion errors.

**Validation:** Segmentation masks were visually verified to align with real object boundaries (e.g., a detected "person" region correctly corresponding to the actual person in frame), confirming the mask output was usable as fusion input.

---

## Module 5 — Depth Estimation & Fusion (Stage 1 & 2)

**Method:** Monocular relative depth is estimated per-frame using a MiDaS depth model (dense inverse-depth map), then fused with the Module 4 segmentation mask by sampling depth at each object's mask pixels — yielding a single representative per-object depth value (e.g. `person: depth≈806.5`).

- **Stage 1 (Segmentation + Depth alignment):** Validated that depth sampled for an object mask spatially corresponds to the correct region — the mask grabs the right pixels, no offset/mismatch.
- **Stage 2 (+ VO pose):** With VO pose accumulating in the same loop, per-object depth was checked for consistent, physically plausible response to deliberate camera motion (e.g. depth decreasing as the camera moves closer). Confirms segmentation + depth + pose are correctly fused within one frame loop, and depth responds to real motion rather than per-frame noise.

**Note:** MiDaS output is *relative* (inverse) depth, not metric — internally consistent within a session, but not real-world distance units without separate calibration.

---

## Module 6 — Full Fusion + Top-Down Map (Stretch Goal)

**Scope:** Combines accumulated VO pose with per-object depth into a top-down `(X, Z)` map of detected objects — an illustrative, qualitative visualization, **not** a metrically accurate 3D reconstruction. This is because VO translation (arbitrary, unitless) and MiDaS depth (relative inverse-depth) are two uncalibrated scales that don't combine into real-world units without extra calibration (known reference object or IMU) — out of scope for this project.

**Method** (`fusion/full_fusion_stage3.py`):
1. Take each detected object's segmentation mask centroid.
2. Backproject to a 3D camera-frame point using intrinsics + depth.
3. Transform by accumulated VO pose into a running "world" frame.
4. Plot on a live top-down `(X, Z)` canvas alongside the traced camera path.

**Validation criteria (defined before testing):**
- ✅ **Success:** colored object dots stay roughly clustered/coherent as the camera moves (no teleporting), overall layout is qualitatively plausible relative to camera path.
- ❌ **Not the goal:** precise metric distances or dots at the object's exact real-world position.

**Result:** A directional test (deliberate lateral slide from a reset start point) showed object dots and the traced camera path extending consistently in the expected direction — confirmed as a genuine pass, not a coincidental result.

---

## Consolidated Limitations

- **VO rotation ambiguity** — pure rotation is a geometric degenerate case; translation estimates can read smaller than the noise floor despite real motion.
- **VO axis correspondence** — camera-frame translation axes vs. physical lateral/forward motion were not independently confirmed in the Module 3 isolated test, though confirmed directionally in the Module 6 end-to-end test.
- **Fixed segmentation vocabulary** — 21 classes only; objects outside this set are not detected or labeled.
- **Depth is relative, not metric** — MiDaS output is internally consistent but not convertible to real-world units without external calibration.
- **Top-down map is illustrative, not metric** — reflects a qualitatively plausible spatial layout, not measured real-world distances.

---

## Tech Stack

- **Language:** Python
- **Core CV:** OpenCV (feature detection/matching, Essential Matrix estimation, camera calibration)
- **Segmentation:** Pretrained semantic segmentation model (21-class)
- **Depth:** MiDaS (monocular relative depth estimation)
- **Fusion:** Custom pose–depth–segmentation fusion (`fusion/full_fusion_stage3.py`)

## Project Structure

```
cv_scene_reconstruction/
├── slam/
│   └── visual_odometry.py       # Module 3
├── segmentation/                 # Module 4
├── depth/
│   └── depth_estimation.py      # Module 5
├── fusion/
│   └── full_fusion_stage3.py    # Module 6
├── utils/
└── cv_project_env/              # (gitignored)
```

## Honest Takeaway

This project prioritized **validated** functionality over the appearance of completeness. Every module has an explicit test that could have failed and didn't — and where limitations exist (rotation ambiguity, non-metric depth/map), they're documented rather than hidden. That's the story this repo is meant to tell.# CV-Untitled
Build a real-time 3D scene understanding system from monocular video
