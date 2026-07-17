"""
main_offset_test.py
-------------------
Runs the inverse projection distance estimation over a dataset sequence
testing multiple bboxBottomOffset values in a single pass.

The bboxBottomOffset shifts the contact point upward by a fraction of the
bbox height before calling calcDistance, while lane geometry and carInLane
checks still use the original bbox bottom (y2).

Usage:
    python main_offset_test.py <sequence>

    <sequence> must be one of:
        waymo10625, waymo10923, waymo11199, waymo13064, waymo13182, kitti15

Output:
    offset_results/results_<sequence>.csv
    One line per frame. Columns: offset_0.00;offset_0.02;...
    Value "-" when no distance was estimated for that frame.

Example:
    python main_offset_test.py waymo10923
    python main_offset_test.py kitti15
"""

import sys
import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from estimation_methods import EstimationMethods, calcDistance
from distance_detector import DistanceDetector
from auxFunctions import resizeFrame

# Offsets to test
OFFSETS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

# Sequence configurations: (f_mm, sensorW_mm, video_filename, isKITTI)
SEQUENCES = {
    "waymo10625": (38.54, 36,    "video_waymo_10625.mp4", False),
    "waymo10923": (38.69, 36,    "video_waymo_10923.mp4", False),
    "waymo11199": (38.99, 36,    "video_waymo_11199.mp4", False),
    "waymo13064": (38.33, 36,    "video_waymo_13064.mp4", False),
    "waymo13182": (38.95, 36,    "video_waymo_13182.mp4", False),
    "kitti15":    (4,     7.469, "video_KITTI_15.mp4",    True),
}

KITTI_PADDING = 75  # pixels added to each side for KITTI width adjustment

# System parameters (same as main.py)
VIDEO_PATH    = "../test_videos/"
MODEL_PATH    = "../models/"
MODEL_NAME    = "v4_2_tasks.onnx"
RESIZED_FRAME = (384, 672)   # (height, width)
CONF_THRESH   = 0.3
IOU_THRESH    = 0.5
TRACKING_IOU  = 0.5
BBOX_MIN_SIZE = 0.025
ROAD_WIDTH    = 3.5
BUFFER_SIZE   = 10
SHOW_WINDOW   = False


def run_sequence(sequence_name):
    if sequence_name not in SEQUENCES:
        print(f"ERROR: sequence '{sequence_name}' not recognised.")
        print(f"Valid options: {list(SEQUENCES.keys())}")
        sys.exit(1)

    f_mm, sensor_w, video_file, is_kitti = SEQUENCES[sequence_name]

    # Output folder
    out_dir = "offset_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{sequence_name}.csv")

    # Load model
    model = YOLO(MODEL_PATH + MODEL_NAME)
    model.predict(source=np.zeros((384, 672, 3), dtype=np.uint8),
                  imgsz=(384, 672), device="cpu")
    print(f"[{sequence_name}] Model loaded")

    # Open video
    vid = cv2.VideoCapture(VIDEO_PATH + video_file)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, first_frame = vid.read()
    if not ret:
        print(f"ERROR: could not open video {VIDEO_PATH + video_file}")
        sys.exit(1)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # KITTI width adjustment (from 1242px to 1392px) - mirrors main.py
    if is_kitti:
        first_frame = cv2.copyMakeBorder(first_frame, 0, 0,
                                         KITTI_PADDING, KITTI_PADDING,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))

    _, padding_top = resizeFrame(first_frame, RESIZED_FRAME)

    # Build detector
    detector = DistanceDetector(
        CONF_THRESH, IOU_THRESH, TRACKING_IOU, BBOX_MIN_SIZE,
        f_mm, sensor_w,
        first_frame.shape, RESIZED_FRAME, padding_top,
        EstimationMethods.inverseProjection,
        ROAD_WIDTH, False,
        {"cars": SHOW_WINDOW, "carId": False, "lanes": SHOW_WINDOW},
        True,
        (0, 255, 0))

    # Per-offset distance buffers and running value (mirror main.py)
    buffers = {off: deque(maxlen=BUFFER_SIZE) for off in OFFSETS}
    current_distance = {off: -1.0 for off in OFFSETS}

    # Write output header
    header = ";".join([f"offset_{off:.2f}" for off in OFFSETS])
    lines = [header]

    total_frames = 0
    print(f"[{sequence_name}] Processing video...")

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        total_frames += 1

        # KITTI width adjustment - mirrors main.py
        if is_kitti:
            frame = cv2.copyMakeBorder(frame, 0, 0,
                                       KITTI_PADDING, KITTI_PADDING,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))

        frame, _ = resizeFrame(frame, RESIZED_FRAME)

        # Run detection + camera parameter estimation
        detector.detectDistances(model, frame)

        # Select the SAME car main.py would. updateDistances() already sorted
        # detector.cars by bbox bottom (nearest first) and stored each car's
        # standard (offset=0) IPM distance, which is None when the car is not
        # in lane. main.py acts on the first car whose distance is truthy:
        #     for car in cars: if car['new']['distance']: ...
        # The carInLane / lane-geometry gating is therefore already baked into
        # that distance, so no separate recheck is needed here.
        target_car = None
        for car in detector.cars:
            if car['new']['distance']:
                target_car = car
                break

        if target_car is None:
            # No car with a valid distance: main.py prints "-" for this frame.
            lines.append(";".join(["-"] * len(OFFSETS)))
        else:
            x1, y1, x2, y2 = target_car['new']['bbox']
            u = x1 + (x2 - x1) / 2

            row_values = []
            for off in OFFSETS:
                v = y2 - int((y2 - y1) * off)   # shift contact point upward

                d = calcDistance(
                    u, v,
                    frame.shape,
                    detector.paddingTop,
                    detector.originalImgH,
                    detector.originalImgW,
                    detector.f_u,
                    detector.f_v,
                    detector.cameraPitch,
                    detector.cameraYaw,
                    detector.cameraHeight
                )

                # Replicate main.py's distance smoothing EXACTLY:
                #  - the first valid (>0) distance is taken directly, NOT buffered
                #  - subsequent valid distances update the sliding-median buffer
                #  - a non-positive/invalid distance leaves the running value
                #    untouched (main.py keeps the previous currentDistance)
                if d is not None and d > 0:
                    if current_distance[off] < 0:
                        current_distance[off] = d
                    else:
                        buffers[off].append(d)
                        current_distance[off] = float(np.median(buffers[off]))

                if current_distance[off] < 0:
                    row_values.append("-")
                else:
                    row_values.append(str(current_distance[off]).replace(".", ","))

            lines.append(";".join(row_values))

        if SHOW_WINDOW:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if total_frames % 50 == 0:
            print(f"  frame {total_frames}")

    vid.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[{sequence_name}] Done. {total_frames} frames -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_offset_test.py <sequence>")
        print(f"  <sequence>: {list(SEQUENCES.keys())}")
        sys.exit(1)

    run_sequence(sys.argv[1])
