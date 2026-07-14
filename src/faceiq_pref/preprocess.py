"""Front-profile face normalization — Python port of the faceiq-labs crop.

Replicates the browser pipeline in faceiq-labs `src/lib/utils/mediapipe.ts`
(`makeHeadshot`), which produced the cohort training images: MediaPipe Face
Landmarker -> eye-level rotation -> head-height square crop (eyes at vertical
center, 12% hair allowance, head fills 65%) -> 1024x1024 on white.

The SageMaker port in faceiq-labs `sagemaker/front-model/code/inference.py`
uses identical geometry; constants below must stay in sync with both.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_SIZE = 1024
HEAD_FILL = 0.65  # head height fills 65% of the crop
EYE_HEIGHT = 0.5  # eye line at vertical center
HAIR_ALLOW = 0.12  # +12% head height for hair

# MediaPipe 478-point mesh indices
MP_LEFT_EYE_OUTER = 33
MP_RIGHT_EYE_OUTER = 263
MP_FOREHEAD = 10
MP_CHIN = 152

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_CACHE = Path(__file__).resolve().parents[2] / "artifacts" / ".cache" / "face_landmarker.task"

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not _MODEL_CACHE.exists():
        _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_CACHE)),
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def normalize_front_photo(img: Image.Image) -> Image.Image | None:
    """faceiq-labs front headshot crop. Returns 1024x1024 RGB, or None if no face."""
    import cv2
    import mediapipe as mp

    rgb = np.array(img.convert("RGB"))
    ih, iw = rgb.shape[:2]

    result = _get_landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]

    le = (lm[MP_LEFT_EYE_OUTER].x * iw, lm[MP_LEFT_EYE_OUTER].y * ih)
    re = (lm[MP_RIGHT_EYE_OUTER].x * iw, lm[MP_RIGHT_EYE_OUTER].y * ih)
    cx, cy = (le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0
    theta = math.atan2(re[1] - le[1], re[0] - le[0])

    # rotate landmarks by -theta around the eye midpoint to measure head height upright
    sin_n, cos_n = math.sin(-theta), math.cos(-theta)

    def rot(px: float, py: float) -> tuple[float, float]:
        dx, dy = px - cx, py - cy
        return cx + dx * cos_n - dy * sin_n, cy + dx * sin_n + dy * cos_n

    chin_r = rot(lm[MP_CHIN].x * iw, lm[MP_CHIN].y * ih)
    fore_r = rot(lm[MP_FOREHEAD].x * iw, lm[MP_FOREHEAD].y * ih)

    head_h = max(8.0, chin_r[1] - fore_r[1])
    if not (head_h > 0 and math.isfinite(head_h)):
        head_h = math.hypot(lm[MP_CHIN].x * iw - cx, lm[MP_CHIN].y * ih - cy) / 0.55

    head_h *= 1.0 + HAIR_ALLOW
    size = min(max(32.0, head_h / HEAD_FILL), max(iw, ih) * 2.0)
    sx = cx - size / 2.0
    sy = cy - EYE_HEIGHT * size

    # rotate the image so the eye line is horizontal, white border fill
    rot_mat = cv2.getRotationMatrix2D((cx, cy), math.degrees(theta), 1.0)
    upright = cv2.warpAffine(
        rgb, rot_mat, (iw, ih),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # extract the square with white fill for out-of-bounds regions
    s = int(round(size))
    x0, y0 = int(round(sx)), int(round(sy))
    canvas = np.full((s, s, 3), 255, dtype=np.uint8)
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x0 + s, iw), min(y0 + s, ih)
    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas[src_y0 - y0 : src_y1 - y0, src_x0 - x0 : src_x1 - x0] = upright[
            src_y0:src_y1, src_x0:src_x1
        ]

    interp = cv2.INTER_AREA if s > OUTPUT_SIZE else cv2.INTER_LINEAR
    out = cv2.resize(canvas, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=interp)
    return Image.fromarray(out)
