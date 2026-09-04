# Gesture Reactive Media System

Real-time facial gesture recognition that pulls double duty as a live-trainable
classifier: look at your webcam, press a key to label your current expression,
and after ~20-30 samples per class you have a working SVM that pops up a
matching image next to your video feed whenever it sees that gesture again.

## How it works

- **Landmark + blendshape extraction** — [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
  gives 478 3D face-mesh landmarks and 52 ARKit-style blendshape scores per
  frame.
- **Hand-engineered geometric features** — eye aspect ratio, brow raise/tilt,
  mouth width/curvature/asymmetry, jaw drop, and a HSV color heuristic over
  the inner-mouth polygon (useful for catching an out-stuck tongue, which
  blendshapes alone don't model well).
- **Classifier** — `StandardScaler -> PCA(0.95 variance) -> RBF SVM`, trained
  on-the-fly from the samples you record.
- **Left/right augmentation** — labels ending in `_left`/`_right` (e.g.
  `wink_left`) get an automatically mirrored synthetic sample for the
  opposite side, roughly doubling the data for those pairs.
- **Temporal smoothing** — predictions are buffered over the last 6 frames
  and require 4-of-6 agreement before the displayed gesture changes, which
  keeps the output from flickering between classes at decision boundaries.

## Setup

```bash
pip install -r requirements.txt
```

Download the blendshape-enabled face landmark model and save it next to the
script as `face_landmarker.task`:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

> Note: this is the same filename as MediaPipe's basic model, but you need
> the blendshape-enabled variant above, or the blendshape features will be
> blank.

Drop images into `assets/`, named after the gesture label, e.g.
`assets/tongue_out.jpg`, `assets/happy.png`, `assets/wink_left.jpg`
(`.jpg`, `.jpeg`, `.png`, `.bmp` all work).

## Usage

```bash
python gesture_reactive_media.py
```

| Key | Action |
|-----|--------|
| `n` | Cycle which gesture label you're about to record samples for |
| `a` | Add a brand-new custom gesture label (typed in the terminal) |
| `r` | Record one training sample of the current label |
| `t` | Train (or retrain) the classifier on everything recorded so far |
| `c` | Clear all recorded samples (keeps the trained model until retrained) |
| `q` | Quit |
