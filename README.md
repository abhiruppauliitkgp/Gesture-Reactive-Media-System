# Gesture Reactive Media System

Real-time facial expression and hand gesture recognition from webcam video, using MediaPipe Tasks landmark models and scikit-learn classifiers, with live-trainable, user-defined gesture-to-media mappings.

## Overview

This project runs two independent, concurrently-executing gesture recognizers on live webcam input:

- **Face recognizer** — classifies facial expressions (e.g. `neutral`, `happy`, `tongue_out`, `wink_left`) from a 71-dimensional feature vector derived from MediaPipe's 478-point face mesh and 52-channel blendshape output.
- **Hand recognizer** — classifies static and directional hand gestures (e.g. `point_left`, `thumbs_up`, `fist`, `peace`) from a 15-dimensional geometric feature vector derived from MediaPipe's 21-point hand landmark model.

Each recognizer is trained live, in-app, on samples the user records from their own webcam, and persists its model to disk. A confidently-detected hand gesture takes display priority over the face gesture; otherwise the face gesture is shown. Detected gestures drive real-time display of a corresponding user-supplied image alongside the webcam feed.

## Features

- **Dual-modality recognition** — face and hand classifiers run independently, each with its own dynamically extensible label set, training buffer, and persisted model.
- **Rich, hand-engineered feature spaces** — not raw landmark coordinates, but scale-normalized geometric ratios, direction vectors, and pretrained blendshape coefficients.
- **Live training loop** — no offline dataset preparation; labels, samples, and retraining all happen interactively via keyboard while the webcam is running.
- **Automatic mirror augmentation** — lateral gesture pairs (`*_left` / `*_right`) are synthetically doubled at training time via geometric reflection.
- **Temporal smoothing** — sliding-window majority voting with a confidence gate suppresses single-frame flicker in the displayed prediction.
- **Landmark visualization overlay** — toggle full mesh/skeleton, key-point-only, or off, to inspect exactly what geometry drives each classification.
- **Pluggable media mapping** — any gesture label maps to an image by filename convention (`assets/<label>.jpg`); no code changes required to add new gesture → image pairs.

## Architecture

```
Webcam frame
   │
   ├─► Face Landmarker (MediaPipe Tasks) ──► 478 landmarks + 52 blendshapes
   │                                              │
   │                                    extract_face_features()
   │                                              │
   │                                     71-dim feature vector
   │                                              │
   │                              StandardScaler → PCA(0.95) → SVM(rbf)
   │                                              │
   │                                    sliding-window majority vote
   │                                              │
   │                                        stable face label
   │
   └─► Hand Landmarker (MediaPipe Tasks) ──► 21 landmarks
                                                    │
                                          extract_hand_features()
                                                    │
                                           15-dim feature vector
                                                    │
                                  StandardScaler → PCA(0.95) → SVM(rbf)
                                                    │
                                          sliding-window majority vote
                                                    │
                                            stable hand label
                                                    │
              hand label (if confident & non-null) else face label
                                                    │
                                       assets/<label>.{jpg,png,...}
                                                    │
                                     displayed beside the video feed
```

## Feature Engineering

### Face (71-dim)

| Group | Dimensions | Description |
|---|---|---|
| Eye aspect ratio | 2 | 6-point EAR per eye |
| Eyebrows | 5 | raise (×2), tilt (×2), inter-brow furrow |
| Mouth | 6 | width, inner/outer opening, curvature, corner asymmetry, jaw drop |
| Face structure | 2 | cheek-width ratio, nose-bridge distance |
| Mouth color | 4 | mean H/S/V + area ratio over the inner-mouth convex hull |
| Blendshapes | 52 | ARKit-standard coefficients from the face landmarker model |

All geometric distances are normalized by an inter-landmark face-scale reference (nose tip to chin), making the feature space invariant to camera distance and face size. The mouth-color descriptor exists because MediaPipe's face mesh has no landmarks on the tongue — geometry alone cannot distinguish an open mouth from a tongue-out gesture, so a color signal from the mouth cavity is used as a complementary feature.

### Hand (15-dim)

| Group | Dimensions | Description |
|---|---|---|
| Finger extension | 4 | tip-to-wrist distance per finger, normalized by palm length |
| Thumb position | 2 | thumb-tip distance to pinky MCP and index MCP |
| Fingertip spread | 3 | adjacent-fingertip distances (index–middle, middle–ring, ring–pinky) |
| Direction vectors | 4 | unit vectors for index-finger orientation and palm orientation (dx, dy each) |
| Aggregate | 2 | mean finger extension, thumb vertical offset |

Directional gestures (`point_left` / `point_right`) are modeled as unit direction vectors rather than static landmark configurations, since the distinguishing signal is orientation, not shape.

## Model

Both recognizers share the same pipeline architecture:

```
StandardScaler → PCA(n_components=0.95) → SVC(kernel="rbf", C=10.0, probability=True)
```

PCA retains 95% of feature variance, controlling for overfitting given the relatively small number of user-recorded samples relative to feature dimensionality. Predictions are smoothed via a 6-frame ring buffer with a 4-of-6 majority-vote threshold and a 0.45 minimum-confidence gate before a prediction is admitted to the buffer.

## Tech Stack

- **Python 3**
- **MediaPipe Tasks API** — `FaceLandmarker`, `HandLandmarker`
- **OpenCV** — video capture, drawing, display
- **scikit-learn** — `StandardScaler`, `PCA`, `SVC`
- **NumPy**
- **joblib** — model persistence

## Installation

```bash
pip install -r requirements.txt
```

Download both MediaPipe task models into the project root:

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

> The face model must be the blendshape-enabled variant (the link above is), or 52 of the 71 face features will be zero.

## Usage

```bash
python gesture_reactive_media.py
```

| Key | Action |
|---|---|
| `m` | Switch active recognizer (face / hand) |
| `n` | Cycle target label |
| `a` | Add a new custom gesture label |
| `r` | Record a training sample for the current label |
| `t` | Train/retrain the active recognizer |
| `c` | Clear the active recognizer's recorded samples |
| `v` | Cycle landmark visualization mode |
| `q` | Quit |

Add corresponding images to `assets/<label>.{jpg,png,bmp}` to have them displayed when a gesture is detected.

## Project Structure

```
.
├── gesture_reactive_media.py       # main application
├── requirements.txt
├── face_landmarker.task            # downloaded, not included
├── hand_landmarker.task            # downloaded, not included
├── face_gestures_config.json       # auto-generated label list
├── hand_gestures_config.json       # auto-generated label list
├── face_gesture_svm_pipeline.pkl   # auto-generated trained model
├── hand_gesture_svm_pipeline.pkl   # auto-generated trained model
└── assets/                         # gesture → image mapping
```

## License

MIT (or your license of choice — update before publishing).
