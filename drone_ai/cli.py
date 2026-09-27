"""
CLI argument parsing and main application entry point.
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from drone_ai.detector import (
    load_model,
    postprocess,
    debug_confidence_stats,
    load_labels,
)
from drone_ai.tracker import SimpleIoUTracker
from drone_ai.drawing import draw_detections
from drone_ai.framebuffer import FramebufferWriter
from drone_ai.camera import Picamera2Wrapper
from drone_ai.guidance import Guidance, GuidanceConfig, configure_logging
from drone_ai.detector import _INTERPRETER_BACKEND

# ---------------------------------------------------------------------------
# Main application logger (detector + pipeline)
# ---------------------------------------------------------------------------
MAIN_LOG = logging.getLogger("detector")


def _configure_main_logging(path: str = "logs/detector.log"):
    """Configure the main detector logger (file + console)."""
    if MAIN_LOG.handlers:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(message)s"
    )
    console = logging.StreamHandler()
    disk = RotatingFileHandler(
        path, maxBytes=5_000_000, backupCount=4, encoding="utf-8"
    )
    for handler in (console, disk):
        handler.setFormatter(formatter)
        MAIN_LOG.addHandler(handler)
    MAIN_LOG.setLevel(logging.INFO)
    MAIN_LOG.propagate = False


def _log(msg: str, level: str = "info"):
    """Convenience wrapper: log to file + print to stdout."""
    getattr(MAIN_LOG, level)(msg)
    print(msg)

# Guidance is optional. If pyserial is missing we degrade gracefully so the
# pure detector pipeline still runs, but print a clear warning.
try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    _SERIAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Global shutdown flag
# ---------------------------------------------------------------------------
_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8-INT8 TFLite detector -> PAL framebuffer overlay "
                    "+ optional marker guidance over MSP"
    )
    p.add_argument("--model", default="config/yolov8n_int8.tflite", help="Path to INT8 TFLite model")
    p.add_argument("--labels", default="config/labels.txt", help="Path to newline-separated class names")
    p.add_argument("--input-size", type=int, default=320, help="Model input size (square)")
    p.add_argument("--infer-every", type=int, default=3,
                   help="Run inference on every Nth captured frame (default 3 = skip 2)")
    p.add_argument("--conf-thresh", type=float, default=0.45,
                   help="Confidence threshold. NOTE: INT8-quantized nano models often top out "
                        "well below 0.6 even on clean detections -- if boxes stop appearing "
                        "entirely, check the [debug] max-confidence log line before raising this")
    p.add_argument("--iou-thresh", type=float, default=0.60,
                   help="NMS IoU threshold (raised default: fewer duplicate/flickering overlapping boxes)")
    p.add_argument("--classes", default="0,2,7",
                   help="Comma-separated COCO class ids to keep, all others discarded "
                        "(default: 0=person,2=car)")
    p.add_argument("--track-iou-thresh", type=float, default=0.05,
                   help="Min IoU to associate a new detection with an existing track between inference cycles")
    p.add_argument("--smooth-alpha", type=float, default=0.3,
                   help="EMA smoothing factor for track box coords: 1.0=snap instantly to new box "
                        "(no smoothing), lower=heavier smoothing/more lag")
    p.add_argument("--track-life", type=int, default=20,
                   help="How many consecutive INFERENCE CYCLES (not display frames) a track is kept "
                        "alive after it stops being re-detected, before it's dropped")
    p.add_argument("--num-threads", type=int, default=4, help="TFLite interpreter threads")
    p.add_argument("--cam-width", type=int, default=720, help="Camera capture width")
    p.add_argument("--cam-height", type=int, default=576, help="Camera capture height")
    p.add_argument("--fb-device", default="/dev/fb0", help="Framebuffer device for composite output")
    p.add_argument("--no-fb", action="store_true",
                   help="Disable framebuffer writes (debug on desktop with a normal window instead)")
    p.add_argument("--log-every", type=int, default=30, help="Log FPS every N frames")
    p.add_argument("--main-log", default="logs/detector.log",
                   help="Path to main detector log file")

    # ---- Guidance-related options (all default to OFF / dry-run) ---------
    g = p.add_argument_group(
        "guidance",
        "Optional marker guidance over MSP. Disabled unless --guidance is set. "
        "READ THE SAFETY WARNING AT THE TOP OF THIS FILE BEFORE USING ANY OF THESE."
    )
    g.add_argument("--guidance-min-confidence", type=float, default=0.35,
                   help="Minimum track score to consider as marker. INT8 nano models "
                        "often top out at 0.5-0.6; set below that.")
    g.add_argument("--guidance-min-track-age", type=int, default=1,
                   help="Inference cycles a track must survive before locking. "
                        "Raise for noise rejection; keep low for fast testing.")
    g.add_argument("--guidance-acquire-timeout", type=float, default=5.0,
                   help="Seconds AUX1 may stay high without a lock before FAULT.")
    g.add_argument("--guidance-max-observation-age", type=float, default=0.5,
                   help="Reject observations older than this. Do NOT raise to hide latency.")
    g.add_argument("--guidance-slew", type=float, default=120.0,
                   help="Max us/sec slew on commanded channels. Raise to test responsiveness.")
    g.add_argument("--guidance", action="store_true",
                   help="Enable the marker-guidance worker (still defaults to dry_run)")
    g.add_argument("--guidance-port", default="/dev/ttyACM0",
                   help="Serial port to flight controller (e.g. /dev/ttyACM0)")
    g.add_argument("--guidance-marker-class", type=int, default=-1,
                   help="Class id of the marker in YOUR model. MUST be a class you "
                        "actually train and pass via --classes as well.")
    g.add_argument("--guidance-log", default="logs/marker_guidance.log",
                   help="Path to guidance log file")
    g.add_argument("--guidance-dry-run", action="store_true", default=True,
                   help="Guidance dry-run (default). Live transmission requires BOTH "
                        "--guidance-live AND --guidance-i-verified-primary-override.")
    g.add_argument("--guidance-live", dest="guidance_dry_run", action="store_false",
                   help="Disable dry-run. Requires --guidance-i-verified-primary-override.")
    g.add_argument("--guidance-i-verified-primary-override", action="store_true",
                   help="Explicit acknowledgement that the Betaflight receiver setup has "
                        "been bench-verified to override ONLY roll/pitch/yaw/throttle via "
                        "MSP, with AUX1 still on the physical receiver. This is a software "
                        "flag only and is NOT a substitute for firmware configuration.")
    g.add_argument("--guidance-raw-order", default="AETR",
                   help="Raw MSP_SET_RAW_RC channel order as observed in Betaflight "
                        "Receiver tab (permutation of AETR). MUST be bench-verified.")
    g.add_argument("--guidance-hover-throttle", type=int, default=1350,
                   help="Manually calibrated hover throttle (us). NOT flight calibration.")
    g.add_argument("--guidance-throttle-min", type=int, default=1250)
    g.add_argument("--guidance-throttle-max", type=int, default=1450)
    g.add_argument("--guidance-desired-box-height", type=float, default=0.25,
                   help="Target normalized box height for the marker (relative cue only)")
    g.add_argument("--guidance-roll-sign", type=int, choices=(-1, 1), default=1)
    g.add_argument("--guidance-pitch-sign", type=int, choices=(-1, 1), default=1)
    g.add_argument("--guidance-yaw-sign", type=int, choices=(-1, 1), default=1)
    g.add_argument("--guidance-throttle-sign", type=int, choices=(-1, 1), default=-1)
    return p.parse_args()


# ===========================================================================
# Guidance builder
# ===========================================================================
def _build_guidance(args, allowed_classes: np.ndarray):
    """
    Validate guidance arguments and construct a Guidance instance, or return
    None if guidance is disabled.
    """
    if not args.guidance:
        return None

    if not _SERIAL_AVAILABLE:
        print("ERROR: --guidance requires pyserial (sudo apt install python3-serial)",
              file=sys.stderr)
        sys.exit(1)
    if not args.guidance_port:
        print("ERROR: --guidance requires --guidance-port", file=sys.stderr)
        sys.exit(1)
    if args.guidance_marker_class < 0:
        print("ERROR: --guidance requires --guidance-marker-class "
              "(the class id of YOUR marker in YOUR model)", file=sys.stderr)
        sys.exit(1)

    # Fail fast if the marker class is not even being kept by the detector
    if args.guidance_marker_class not in allowed_classes.tolist():
        print(f"ERROR: --guidance-marker-class {args.guidance_marker_class} is not in "
              f"--classes {allowed_classes.tolist()}; the detector will never emit it",
              file=sys.stderr)
        sys.exit(1)

    if not args.guidance_dry_run and not args.guidance_i_verified_primary_override:
        print("ERROR: --guidance-live requires --guidance-i-verified-primary-override. "
              "Read the safety warning at the top of this file.", file=sys.stderr)
        sys.exit(1)

    configure_logging(args.guidance_log)

    cfg = GuidanceConfig(
        port=args.guidance_port,
        marker_class_id=args.guidance_marker_class,
        raw_order=args.guidance_raw_order,
        hover_throttle=args.guidance_hover_throttle,
        throttle_min=args.guidance_throttle_min,
        throttle_max=args.guidance_throttle_max,
        dry_run=args.guidance_dry_run,
        primary_override_verified=args.guidance_i_verified_primary_override,
        desired_box_height=args.guidance_desired_box_height,
        roll_sign=args.guidance_roll_sign,
        pitch_sign=args.guidance_pitch_sign,
        yaw_sign=args.guidance_yaw_sign,
        throttle_sign=args.guidance_throttle_sign,
        minimum_confidence=args.guidance_min_confidence,
        minimum_track_age=args.guidance_min_track_age,
        acquire_timeout=args.guidance_acquire_timeout,
        max_observation_age=args.guidance_max_observation_age,
        slew_us_per_second=args.guidance_slew,
    )
    return Guidance(cfg)


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    # Configure main pipeline logging to file + console
    _configure_main_logging(args.main_log)

    _log(f"[init] TFLite backend: {_INTERPRETER_BACKEND}")
    labels = load_labels(args.labels)

    # ---- Load model -------------------------------------------------------
    interpreter, input_details, output_details = load_model(
        args.model, args.input_size, args.num_threads
    )

    in_dtype = input_details[0]["dtype"]
    in_scale, in_zero_point = input_details[0].get("quantization", (0.0, 0))
    out_scale, out_zero_point = output_details[0].get("quantization", (0.0, 0))

    # Read the model's ACTUAL input shape
    in_shape = list(input_details[0]["shape"])
    if len(in_shape) != 4:
        print(f"ERROR: unexpected input tensor rank {in_shape}, expected 4D", file=sys.stderr)
        sys.exit(1)

    if in_shape[1] == 3:
        input_layout = "NCHW"
        model_h, model_w = in_shape[2], in_shape[3]
    elif in_shape[3] == 3:
        input_layout = "NHWC"
        model_h, model_w = in_shape[1], in_shape[2]
    else:
        print(f"ERROR: could not infer input layout from shape {in_shape}", file=sys.stderr)
        sys.exit(1)

    if (model_h, model_w) != (args.input_size, args.input_size):
        print(f"[init] note: model expects {model_w}x{model_h}, overriding --input-size "
              f"({args.input_size})", file=sys.stderr)
        args.input_size = model_h  # assume square input, as documented

    out_shape = output_details[0]["shape"]
    num_classes = (out_shape[1] - 4) if out_shape[1] < out_shape[2] else (out_shape[2] - 4)

    allowed_classes = np.array(sorted({int(c) for c in args.classes.split(",") if c.strip() != ""}),
                                dtype=np.int32)

    _log(f"[init] Model loaded ({args.model}), input {args.input_size}x{args.input_size}, "
          f"layout={input_layout}, threads={args.num_threads}, quantized_input={in_dtype}")
    _log(f"[init] Keeping classes {allowed_classes.tolist()} "
          f"(conf>={args.conf_thresh}, nms_iou<={args.iou_thresh}); "
          f"tracker: match_iou>={args.track_iou_thresh}, smooth_alpha={args.smooth_alpha}, "
          f"track_life={args.track_life} inference cycles")

    # Preallocate the model input tensor buffer once
    model_input = np.empty(tuple(in_shape), dtype=in_dtype)

    # ---- Guidance (validate/construct BEFORE opening camera/framebuffer) --
    guidance = _build_guidance(args, allowed_classes)

    # ---- Camera -------------------------------------------------------
    picam2 = Picamera2Wrapper()
    picam2.start(args.cam_width, args.cam_height)
    time.sleep(1.0)  # sensor warm-up
    _log(f"[init] Camera started at {args.cam_width}x{args.cam_height}")

    # ---- Framebuffer / PAL output -----------------------------------------
    fb_writer = None
    if not args.no_fb:
        try:
            fb_writer = FramebufferWriter(args.fb_device, args.cam_width, args.cam_height)
            _log(f"[init] Writing composite output to {args.fb_device} "
                  f"({fb_writer.bpp} bpp)")
        except OSError as e:
            _log(f"[warn] could not open {args.fb_device} ({e}); "
                  f"falling back to on-screen window", "warning")
            fb_writer = None

    tracker = SimpleIoUTracker(
        iou_match_thresh=args.track_iou_thresh,
        smooth_alpha=args.smooth_alpha,
        max_misses=args.track_life,
    )
    last_infer_ms = 0.0

    # Start the guidance worker only now, after camera+fb are up
    if guidance is not None:
        guidance.start()
        mode = "DRY-RUN" if args.guidance_dry_run else "LIVE"
        _log(f"[init] Guidance started ({mode}); marker_class="
              f"{args.guidance_marker_class}, port={args.guidance_port}")

    frame_count = 0
    infer_count = 0
    t_fps_window_start = time.time()
    frames_in_window = 0

    _log("[run] entering main loop, Ctrl+C to stop")

    try:
        while _running:
            # Timestamp taken BEFORE capture_array() so the guidance module
            # accounts for capture-call duration as part of observation age.
            capture_started = time.monotonic()
            frame_rgb = picam2.capture_array()  # RGB888, no extra copy
            frame_count += 1

            if frame_count % args.infer_every == 0:
                t0 = time.time()

                # Resize + (if needed) requantize into the preallocated buffer
                resized = picam2.resize_frame(frame_rgb, args.input_size)

                if input_layout == "NCHW":
                    prepared = np.transpose(resized, (2, 0, 1))  # HWC -> CHW
                else:
                    prepared = resized

                if in_dtype in (np.uint8, np.int8):
                    prepared = prepared.astype(in_dtype)
                else:
                    prepared = (prepared.astype(np.float32) / 255.0)

                np.copyto(model_input[0], prepared)

                interpreter.set_tensor(input_details[0]["index"], model_input)
                interpreter.invoke()
                raw_output = interpreter.get_tensor(output_details[0]["index"])

                if out_scale:
                    raw_output = (raw_output.astype(np.float32) - out_zero_point) * out_scale

                if infer_count < 5 or infer_count % 15 == 0:
                    _log(f"[debug] pre-threshold confidence: "
                          f"{debug_confidence_stats(raw_output, num_classes, allowed_classes)}")

                boxes, scores, class_ids = postprocess(
                    raw_output, args.conf_thresh, args.iou_thresh, num_classes,
                    allowed_classes=allowed_classes,
                )
                # One tracker update per INFERENCE CYCLE
                tracker.update(boxes, scores, class_ids)
                last_infer_ms = (time.time() - t0) * 1000.0
                infer_count += 1

                # Submit ONLY on inference cycles
                if guidance is not None:
                    guidance.submit_tracks(
                        tracker.get_display_tracks(),
                        captured_at=capture_started,
                    )

            # Draw every currently-alive track on every frame (fresh or coasting)
            active_tracks = tracker.get_display_tracks()
            frame_bgr = picam2.rgb_to_bgr(frame_rgb)
            draw_detections(frame_bgr, active_tracks, labels, args.cam_width, args.cam_height)

            if fb_writer is not None:
                fb_writer.write(frame_bgr)
            else:
                picam2.show_window(frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames_in_window += 1
            if frame_count % args.log_every == 0:
                elapsed = time.time() - t_fps_window_start
                fps = frames_in_window / elapsed if elapsed > 0 else 0.0
                avg_conf = float(np.mean([t.score for t in active_tracks])) if active_tracks else 0.0
                coasting = sum(1 for t in active_tracks if t.misses > 0)
                _log(f"[stats] display_fps={fps:.1f} last_infer_ms={last_infer_ms:.1f} "
                      f"infer_count={infer_count} active_tracks={len(active_tracks)} "
                      f"(coasting={coasting}) avg_conf={avg_conf:.2f}")
                t_fps_window_start = time.time()
                frames_in_window = 0

    finally:
        _log("[shutdown] stopping guidance...")
        if guidance is not None:
            try:
                guidance.close()
            except Exception:
                _log("[shutdown] guidance.close() raised", "error")

        _log("[shutdown] stopping camera and cleaning up...")
        try:
            picam2.stop()
        except Exception:
            pass
        if fb_writer is not None:
            fb_writer.close()
        else:
            picam2.destroy_windows()
        _log("[shutdown] done")


if __name__ == "__main__":
    main()
