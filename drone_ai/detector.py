"""
Model loading, inference, and post-processing (NMS, filtering).
"""

import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TFLite interpreter import: prefer ai_edge_litert, fall back to tflite_runtime
# ---------------------------------------------------------------------------
_INTERPRETER_BACKEND = None
try:
    from ai_edge_litert.interpreter import Interpreter  # newer Google package
    _INTERPRETER_BACKEND = "ai_edge_litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        _INTERPRETER_BACKEND = "tflite_runtime"
    except ImportError:
        print(
            "ERROR: neither ai_edge_litert nor tflite_runtime is installed.\n"
            "  pip install ai-edge-litert   (preferred)\n"
            "  pip install tflite-runtime   (fallback)",
            file=sys.stderr,
        )
        sys.exit(1)


def load_model(model_path: str, input_size: int, num_threads: int):
    """
    Load a TFLite model and return (interpreter, input_details, output_details).

    Auto-detects input layout (NHWC vs NCHW) and overrides input_size if the
    model expects a different dimension.
    """
    interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Validate input tensor shape
    in_shape = list(input_details[0]["shape"])
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected input tensor rank {in_shape}, expected 4D")

    if in_shape[1] == 3:
        input_layout = "NCHW"
        model_h, model_w = in_shape[2], in_shape[3]
    elif in_shape[3] == 3:
        input_layout = "NHWC"
        model_h, model_w = in_shape[1], in_shape[2]
    else:
        raise ValueError(f"Could not infer input layout from shape {in_shape}")

    if (model_h, model_w) != (input_size, input_size):
        logger.info(
            "Model expects %dx%d, overriding input-size (%d)",
            model_w, model_h, input_size,
        )
        input_size = model_h  # assume square input

    logger.info(
        "Model loaded (%s), input %dx%d, layout=%s, threads=%d",
        model_path, input_size, input_size, input_layout, num_threads,
    )

    return interpreter, input_details, output_details


# ---------------------------------------------------------------------------
# Post-processing: vectorized NMS
# ---------------------------------------------------------------------------
def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """Convert center-x, center-y, width, height to x1, y1, x2, y2."""
    x, y, w, h = (
        boxes_xywh[:, 0],
        boxes_xywh[:, 1],
        boxes_xywh[:, 2],
        boxes_xywh[:, 3],
    )
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    Standard greedy NMS, vectorized IoU computation. Returns kept indices.
    """
    if boxes_xyxy.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = (
        boxes_xyxy[:, 0],
        boxes_xyxy[:, 1],
        boxes_xyxy[:, 2],
        boxes_xyxy[:, 3],
    )
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

        order = rest[iou <= iou_thresh]

    return np.array(keep, dtype=np.int64)


def postprocess(
    raw_output: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    num_classes: int,
    allowed_classes: np.ndarray = None,
):
    """
    Expects YOLOv8-style output of shape (1, 4+num_classes, num_anchors) or
    (1, num_anchors, 4+num_classes); handles both layouts.

    Returns (boxes_xyxy_norm[N,4], scores[N], class_ids[N]) in normalized [0,1] coords.

    allowed_classes: if given, detections whose class_id is not in this array
    are dropped BEFORE NMS. Filtering early means NMS only has to consider
    the allowed classes, which is both cheaper and prevents a high-confidence
    unwanted class from suppressing a wanted one via IoU.
    """
    out = np.squeeze(raw_output)
    if out.shape[0] == 4 + num_classes:
        out = out.T  # -> (num_anchors, 4+num_classes)

    boxes_xywh = out[:, :4]
    class_scores = out[:, 4:4 + num_classes]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_thresh
    if allowed_classes is not None:
        mask &= np.isin(class_ids, allowed_classes)

    if not np.any(mask):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms(boxes_xyxy, scores, iou_thresh)

    return boxes_xyxy[keep], scores[keep], class_ids[keep]


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------
def debug_confidence_stats(raw_output: np.ndarray, num_classes: int, allowed_classes: np.ndarray):
    """
    Computes the raw max confidence per allowed class BEFORE any thresholding,
    so you can see what the model is actually producing regardless of
    --conf-thresh.

    Helps distinguish three failure modes:
      1. conf_thresh too high for the model's real output range
      2. class filter / index mismatch (custom-trained model)
      3. genuinely no target in frame / camera or model input problem
    """
    out = np.squeeze(raw_output)
    if out.shape[0] == 4 + num_classes:
        out = out.T
    class_scores = out[:, 4:4 + num_classes]
    per_class_max = class_scores.max(axis=0)
    overall_best_class = int(np.argmax(per_class_max))
    allowed_str = ", ".join(
        f"class{c}={per_class_max[c]:.3f}"
        for c in allowed_classes.tolist()
        if 0 <= c < num_classes
    )
    return (
        f"{allowed_str} | overall_best=class{overall_best_class}"
        f"({per_class_max[overall_best_class]:.3f})"
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def load_labels(path: str) -> list:
    """Load newline-separated class names from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        logger.warning("Could not read labels file '%s', using numeric class ids", path)
        return []
