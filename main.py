"""
Gesture Reactive Media System
==============================
Detects facial gestures via a rich set of MediaPipe Face Landmarker features
(478 3D landmarks + 52 ARKit-style blendshape scores + a mouth-interior color
heuristic for tongue detection) and displays a matching image next to your
webcam feed.

Setup
-----
1. pip install -r requirements.txt
2. Download the blendshape-enabled face landmark model:
   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
   Save it next to this script as `face_landmarker.task`.
   (This is the same filename as the basic model, but you MUST use the
   version that supports blendshapes or blendshape features will be blank.)
3. Put images in an `assets/` folder named after the gesture, e.g.
   assets/tongue_out.jpg, assets/happy.png, assets/wink_left.jpg
   Any of .jpg/.jpeg/.png/.bmp works; the first match is used.
4. Run: python gesture_reactive_media.py

Controls
--------
  n : cycle which gesture label you are about to record samples for
  a : add a brand-new custom gesture label (type it in the terminal)
  r : record one training sample of the current label from your current expression
  t : train (or retrain) the classifier on everything recorded so far
  c : clear all recorded samples (keeps trained model until you retrain)
  q : quit

Tips for accuracy
------------------
  - Record 40-80 samples per gesture, moving your head slightly and varying
    lighting a little between samples, so the model doesn't overfit to one pose.
  - Always record a generous number of "neutral" samples - it's the class
    everything else is being distinguished from.
  - For left/right specific gestures (wink_left, wink_right, smirk_left...)
    the script automatically also synthesizes a mirrored sample for the
    opposite label, so you get roughly double the data for those pairs.
  - Retrain (t) any time after adding more samples.
"""

import os
import json
import glob
from collections import deque

import cv2
import numpy as np
import joblib
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
# MediaPipe setup
# --------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"
PIPELINE_PATH = "gesture_svm_pipeline.pkl"
CONFIG_PATH = "gestures_config.json"
ASSETS_DIR = "assets"

DEFAULT_CLASSES = [
    "neutral", "happy", "sad", "angry", "surprised",
    "tongue_out",
]

# The fixed ARKit blendshape category order MediaPipe's FaceLandmarker
# returns them in. Building an explicit name->value lookup (instead of
# assuming index order) keeps features stable even if the runtime ever
# reorders results.
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

# Pairs of blendshape/feature names that must swap when we mirror a sample
# (used for left/right label augmentation).
BLENDSHAPE_MIRROR_PAIRS = [
    ("browDownLeft", "browDownRight"), ("browOuterUpLeft", "browOuterUpRight"),
    ("cheekSquintLeft", "cheekSquintRight"), ("eyeBlinkLeft", "eyeBlinkRight"),
    ("eyeLookDownLeft", "eyeLookDownRight"), ("eyeLookInLeft", "eyeLookInRight"),
    ("eyeLookOutLeft", "eyeLookOutRight"), ("eyeLookUpLeft", "eyeLookUpRight"),
    ("eyeSquintLeft", "eyeSquintRight"), ("eyeWideLeft", "eyeWideRight"),
    ("jawLeft", "jawRight"), ("mouthDimpleLeft", "mouthDimpleRight"),
    ("mouthFrownLeft", "mouthFrownRight"), ("mouthLowerDownLeft", "mouthLowerDownRight"),
    ("mouthPressLeft", "mouthPressRight"), ("mouthLeft", "mouthRight"),
    ("mouthSmileLeft", "mouthSmileRight"), ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthUpperUpLeft", "mouthUpperUpRight"), ("noseSneerLeft", "noseSneerRight"),
]

# --------------------------------------------------------------------------
# Landmark indices (MediaPipe 478-point face mesh topology)
# --------------------------------------------------------------------------
NOSE_TIP, NOSE_BRIDGE, CHIN = 1, 6, 152

LEFT_EYE_6 = [33, 160, 158, 133, 153, 144]      # outer,top1,top2,inner,bot2,bot1
RIGHT_EYE_6 = [263, 387, 385, 362, 380, 373]

LEFT_BROW_MID, LEFT_BROW_INNER, LEFT_BROW_OUTER = 105, 55, 65
RIGHT_BROW_MID, RIGHT_BROW_INNER, RIGHT_BROW_OUTER = 334, 285, 295

MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT = 61, 291
INNER_LIP_UPPER, INNER_LIP_LOWER = 13, 14
OUTER_LIP_UPPER, OUTER_LIP_LOWER = 0, 17
INNER_CORNER_LEFT, INNER_CORNER_RIGHT = 78, 308

CHEEK_LEFT, CHEEK_RIGHT = 50, 280
FACE_LEFT_EDGE, FACE_RIGHT_EDGE = 234, 454

# Rough inner-mouth cavity polygon, used only for the color heuristic.
INNER_MOUTH_POLY = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                     324, 318, 402, 317, 14, 87, 178, 88, 95]

FEATURE_NAMES = [
    "left_ear", "right_ear",
    "left_brow_raise", "right_brow_raise", "brow_furrow",
    "left_brow_tilt", "right_brow_tilt",
    "mouth_width", "inner_mouth_open", "outer_mouth_open",
    "mouth_curvature", "mouth_corner_asym", "jaw_drop",
    "cheek_width_ratio", "nose_wrinkle",
    "mouth_hue", "mouth_sat", "mouth_val", "mouth_area_ratio",
]


def _dist(p1, p2, w, h):
    return float(np.hypot((p2.x - p1.x) * w, (p2.y - p1.y) * h))


def _eye_aspect_ratio(idxs, lm, w, h):
    p = [lm[i] for i in idxs]
    vertical = _dist(p[1], p[5], w, h) + _dist(p[2], p[4], w, h)
    horizontal = _dist(p[0], p[3], w, h) + 1e-6
    return vertical / (2.0 * horizontal)


def _mouth_color_features(frame_bgr, lm, w, h):
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in INNER_MOUTH_POLY])
    x, y, bw, bh = cv2.boundingRect(pts)
    x, y = max(x, 0), max(y, 0)
    bw, bh = max(bw, 1), max(bh, 1)
    if x + bw > w or y + bh > h or bw < 4 or bh < 4:
        return 0.0, 0.0, 0.0, 0.0
    mask = np.zeros((bh, bw), dtype=np.uint8)
    shifted = pts - [x, y]
    cv2.fillConvexPoly(mask, cv2.convexHull(shifted), 255)
    roi = frame_bgr[y:y + bh, x:x + bw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(mask) == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_h, mean_s, mean_v = cv2.mean(hsv, mask=mask)[:3]
    area_ratio = cv2.countNonZero(mask) / float(w * h)
    return float(mean_h), float(mean_s), float(mean_v), float(area_ratio)


def extract_features(landmarks, frame_bgr, w, h):
    """Rich geometric feature vector from face-mesh landmarks + mouth color."""
    lm = landmarks
    face_scale = _dist(lm[NOSE_TIP], lm[CHIN], w, h) or 1.0

    left_ear = _eye_aspect_ratio(LEFT_EYE_6, lm, w, h)
    right_ear = _eye_aspect_ratio(RIGHT_EYE_6, lm, w, h)

    left_brow_raise = (lm[LEFT_BROW_MID].y - lm[LEFT_EYE_6[1]].y) / face_scale
    right_brow_raise = (lm[RIGHT_BROW_MID].y - lm[RIGHT_EYE_6[1]].y) / face_scale
    brow_furrow = _dist(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER], w, h) / face_scale
    left_brow_tilt = (lm[LEFT_BROW_INNER].y - lm[LEFT_BROW_OUTER].y) / face_scale
    right_brow_tilt = (lm[RIGHT_BROW_INNER].y - lm[RIGHT_BROW_OUTER].y) / face_scale

    mouth_width = _dist(lm[MOUTH_CORNER_LEFT], lm[MOUTH_CORNER_RIGHT], w, h) / face_scale
    inner_mouth_open = _dist(lm[INNER_LIP_UPPER], lm[INNER_LIP_LOWER], w, h) / face_scale
    outer_mouth_open = _dist(lm[OUTER_LIP_UPPER], lm[OUTER_LIP_LOWER], w, h) / face_scale
    mouth_corners_y = (lm[MOUTH_CORNER_LEFT].y + lm[MOUTH_CORNER_RIGHT].y) / 2.0
    mouth_curvature = (mouth_corners_y - lm[INNER_LIP_UPPER].y) / face_scale
    mouth_corner_asym = (lm[MOUTH_CORNER_LEFT].y - lm[MOUTH_CORNER_RIGHT].y) / face_scale

    jaw_drop = _dist(lm[OUTER_LIP_LOWER], lm[CHIN], w, h) / face_scale
    cheek_width_ratio = _dist(lm[CHEEK_LEFT], lm[CHEEK_RIGHT], w, h) / (
        _dist(lm[FACE_LEFT_EDGE], lm[FACE_RIGHT_EDGE], w, h) + 1e-6)
    nose_wrinkle = _dist(lm[NOSE_BRIDGE], lm[NOSE_TIP], w, h) / face_scale

    mouth_hue, mouth_sat, mouth_val, mouth_area_ratio = _mouth_color_features(frame_bgr, lm, w, h)

    return np.array([
        left_ear, right_ear,
        left_brow_raise, right_brow_raise, brow_furrow,
        left_brow_tilt, right_brow_tilt,
        mouth_width, inner_mouth_open, outer_mouth_open,
        mouth_curvature, mouth_corner_asym, jaw_drop,
        cheek_width_ratio, nose_wrinkle,
        mouth_hue, mouth_sat, mouth_val, mouth_area_ratio,
    ], dtype=np.float64)


def extract_blendshapes(face_blendshapes):
    """Fixed-order 52-length blendshape vector; zeros if unavailable."""
    if not face_blendshapes:
        return np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float64)
    scores = {c.category_name: c.score for c in face_blendshapes[0]}
    return np.array([scores.get(name, 0.0) for name in BLENDSHAPE_NAMES], dtype=np.float64)


def full_feature_vector(landmarks, face_blendshapes, frame_bgr, w, h):
    geo = extract_features(landmarks, frame_bgr, w, h)
    bs = extract_blendshapes(face_blendshapes)
    return np.concatenate([geo, bs])


def mirror_feature_vector(vec):
    """Flip a feature vector left<->right, for data augmentation."""
    v = vec.copy()
    n_geo = len(FEATURE_NAMES)

    # Geometric side-swaps (indices refer to FEATURE_NAMES order above).
    swap_pairs = [(0, 1), (2, 3), (5, 6)]  # ear, brow_raise, brow_tilt
    for a, b in swap_pairs:
        v[a], v[b] = v[b], v[a]
    v[11] = -v[11]  # mouth_corner_asym flips sign when mirrored

    bs_offset = n_geo
    name_to_idx = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}
    for left_name, right_name in BLENDSHAPE_MIRROR_PAIRS:
        li, ri = bs_offset + name_to_idx[left_name], bs_offset + name_to_idx[right_name]
        v[li], v[ri] = v[ri], v[li]
    return v


def mirror_label(label):
    if label.endswith("_left"):
        return label[:-5] + "_right"
    if label.endswith("_right"):
        return label[:-6] + "_left"
    return label


# --------------------------------------------------------------------------
# Config / assets persistence
# --------------------------------------------------------------------------
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            classes = data.get("classes", DEFAULT_CLASSES)
            if classes:
                return classes
        except Exception:
            pass
    return list(DEFAULT_CLASSES)


def save_config(classes):
    with open(CONFIG_PATH, "w") as f:
        json.dump({"classes": classes}, f, indent=2)


def find_gesture_image(label):
    for ext in ("jpg", "jpeg", "png", "bmp"):
        matches = glob.glob(os.path.join(ASSETS_DIR, f"{label}.{ext}"))
        if matches:
            return matches[0]
    return None


def load_gesture_image(label, target_h, target_w):
    path = find_gesture_image(label)
    if path:
        img = cv2.imread(path)
        if img is not None:
            return cv2.resize(img, (target_w, target_h))
    blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    cv2.putText(blank, f"No image for:", (20, target_h // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(blank, f"'{label}'", (20, target_h // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(blank, f"(add assets/{label}.jpg)", (20, target_h // 2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return blank


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)),
    ])


def augment_with_mirrors(X, y):
    """Doubles up left/right-specific samples with mirrored synthetic ones."""
    X_aug, y_aug = list(X), list(y)
    for vec, label in zip(X, y):
        mirrored_label = mirror_label(label)
        if mirrored_label != label:
            X_aug.append(mirror_feature_vector(vec))
            y_aug.append(mirrored_label)
    return X_aug, y_aug


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: missing '{MODEL_PATH}'. See the docstring at the top of this file "
              f"for the download link (make sure it's the blendshape-enabled model).")
        return

    os.makedirs(ASSETS_DIR, exist_ok=True)

    gesture_classes = load_config()
    current_class_idx = 0

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    X_data, y_data = [], []
    pipeline = None
    is_trained = False

    prediction_buffer = deque(maxlen=6)
    stable_gesture = "neutral"

    if os.path.exists(PIPELINE_PATH):
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            is_trained = True
            print(f"Loaded existing model from '{PIPELINE_PATH}'.")
        except Exception as e:
            print(f"Could not load saved model: {e}")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            features = None
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                features = full_feature_vector(
                    landmarks, detection_result.face_blendshapes, frame, w, h)

                if is_trained and pipeline is not None:
                    probs = pipeline.predict_proba([features])[0]
                    max_idx = int(np.argmax(probs))
                    max_prob = probs[max_idx]
                    pred_class = pipeline.classes_[max_idx]

                    prediction_buffer.append(pred_class if max_prob >= 0.45 else stable_gesture)

                    classes, counts = np.unique(prediction_buffer, return_counts=True)
                    best = classes[np.argmax(counts)]
                    if counts[np.argmax(counts)] >= 4:  # 4-of-6 agreement -> stable
                        stable_gesture = best
            else:
                prediction_buffer.append("neutral")

            display_img = load_gesture_image(stable_gesture, h, w)

            target_label = gesture_classes[current_class_idx]
            cv2.putText(frame, f"Target Label [n/a]: {target_label.upper()}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            status_str = f"Detected: {stable_gesture.upper()}" if is_trained else "RECORDING / NOT TRAINED YET"
            color = (0, 255, 0) if is_trained else (0, 0, 255)
            cv2.putText(frame, status_str, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"Samples: {len(X_data)}  Classes: {len(set(y_data)) if y_data else 0}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, "n:next  a:add-label  r:record  t:train  c:clear  q:quit",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            combined = cv2.hconcat([frame, display_img])
            cv2.imshow("Gesture Reactive Media System", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                current_class_idx = (current_class_idx + 1) % len(gesture_classes)
            elif key == ord("a"):
                new_label = input("Type the new gesture's label (letters/numbers/underscore): ").strip()
                new_label = new_label.lower().replace(" ", "_")
                if new_label and new_label not in gesture_classes:
                    gesture_classes.append(new_label)
                    save_config(gesture_classes)
                    current_class_idx = len(gesture_classes) - 1
                    print(f"Added new gesture '{new_label}'. Drop an image at "
                          f"assets/{new_label}.jpg to have it display when detected.")
            elif key == ord("r"):
                if features is not None:
                    X_data.append(features)
                    y_data.append(target_label)
                    print(f"Recorded sample for '{target_label}'. Total: {len(X_data)}")
                else:
                    print("No face detected - sample not recorded.")
            elif key == ord("c"):
                X_data, y_data = [], []
                print("Cleared all recorded samples.")
            elif key == ord("t"):
                if len(set(y_data)) >= 2:
                    X_train, y_train = augment_with_mirrors(X_data, y_data)
                    pipeline = build_pipeline()
                    pipeline.fit(X_train, y_train)
                    is_trained = True
                    joblib.dump(pipeline, PIPELINE_PATH)
                    print(f"\nTrained on {len(X_train)} samples "
                          f"({len(X_data)} recorded + {len(X_train) - len(X_data)} mirrored) "
                          f"across {len(set(y_train))} classes. Saved to '{PIPELINE_PATH}'.\n")
                else:
                    print("Need at least 2 distinct labeled classes with samples before training.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()