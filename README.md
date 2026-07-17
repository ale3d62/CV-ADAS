# Longitudinal distance estimation using monocular vision and geometric autocalibration

Source code of the Master's Thesis *"Longitudinal distance estimation in traffic videos using
monocular vision and geometric autocalibration"*.

From a single camera (dashcam), the system detects the vehicles and lanes on the road, autocalibrates
the camera (height, *pitch* and *yaw*) and estimates the distance to the leading vehicle using two
geometric methods: lane-width estimation and inverse projection. It also includes the scripts used to
prepare the test videos and obtain the reference values (*ground truth*) from the KITTI and Waymo
datasets.

![preview](https://github.com/ale3d62/CV-ADAS/blob/TFM/readmegif.gif?raw=true)

---

## 1. Repository structure

| Folder | Contents |
|---|---|
| `src/` | System source code. Contains the two executable components: `main.py` and `main_offset_test.py`. |
| `models/` | Detection models used by the system. |
| `test_videos/` | Input videos on which the system is run. Due to submission size limits (100 MB max), only 3 of the 6 sequences used are included. |
| `datasets/` | Dataset preprocessing and ground-truth extraction scripts, together with one sample sequence from KITTI and one from Waymo. |

---

## 2. Requirements and installation

The system was developed and tested with **Python 3.7.9**. More recent versions may work, but this is
not guaranteed.

Dependencies are listed in `requirements.txt` files. Using a virtual environment is recommended:

```bash
# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Detection system dependencies
pip install -r src/requirements.txt

# Dataset scripts dependencies (only if they are going to be used)
pip install -r datasets/requirements.txt
```

---

## 3. Usage

The two executable components are located in `src/` and must be launched from that folder.

### 3.1. `main.py` — Detection and distance estimation

Processes a video and displays, frame by frame, the detected vehicles and lanes together with the
estimated distance to the leading vehicle. When finished, it prints execution-time statistics and CPU
and memory usage.

```bash
cd src
python main.py
```

Configuration is done by editing the variables in the top block of the file:

| Parameter | Description |
|---|---|
| `sequence` | Video sequence to process (one of the predefined `SequenceConfig` entries). |
| `modelName` / `modelPath` | Detection model and its path. |
| `resizedFrameSize` | Working resolution `(height, width)`. |
| `yoloConfThresh`, `yoloIouThresh`, `trackingIouThresh` | Confidence, NMS and *tracking* thresholds. |
| `bBoxMinSize` | Minimum *bbox* size (fraction of the image) to consider a vehicle. |
| `estimationMethod` | Estimation method: `roadWidthEstimation` (lane width) or `inverseProjection` (inverse projection). |
| `roadWidth` | Assumed lane width (m). |
| `heightCorrection` | Applies the estimated camera-height correction (lane-width method only). |
| `distanceBufferSize` | Size of the temporal distance-filter buffer. |
| `filterCarInLane` | If enabled, only estimates the distance to the vehicle in the ego lane. |
| `showSettings` | Elements to display in the visualization: vehicles, identifiers and lanes. |
| `showTime`, `showCPUUsage`, `showMemoryUsage` | Enable the final performance statistics. |
| `dataExtractionType` | Value to print to the console (one per frame) for comparison against the *ground truth*: camera distance, height, *pitch* or *yaw*. |

### 3.2. `main_offset_test.py` — Bounding-box offset test

Evaluates the accuracy of the inverse-projection estimation against different offsets of the vehicle's
ground-contact point. It takes the sequence as an argument and generates a CSV file with the results in
the `offset_results/` folder.

```bash
cd src
python main_offset_test.py <sequence>
```

Where `<sequence>` is one of: `waymo10625`, `waymo10923`, `waymo11199`, `waymo13064`, `waymo13182`,
`kitti15`. For example:

```bash
python main_offset_test.py waymo10923
```

---

## 4. Datasets

The system works on `.mp4` videos, which are generated from the original datasets. The scripts in the
`datasets/` folder allow generating these videos and obtaining the reference values. The submission
includes one sample sequence per dataset; to reproduce the remaining tests, download the corresponding
sequences and place them in their folders.

### 4.1. KITTI

Download: KITTI Raw Data — <https://www.cvlibs.net/datasets/kitti/raw_data.php>. The sequences are
placed inside `datasets/kitti_dataset/`, keeping KITTI's original organization (synced and rectified
data, calibration and *tracklets*).

Scripts (`datasets/kitti_dataset/`):

- **`kitti_distance_extractor.py`** — Real longitudinal distance to the leading vehicle, frame by frame.
- **`kitti_rotation_extractor.py`** — Real *pitch* and *yaw* values of the camera relative to the road.

### 4.2. Waymo

Download: Waymo Open Dataset — *Perception dataset (v2, modular `.parquet` format)* —
<https://waymo.com/open/download/>. For each sequence you need the `camera_image`, `camera_calibration`
and `lidar_box` components, which are placed in the `camera_images/`, `camera_calibration/` and
`lidar_box/` folders of `datasets/waymo_dataset/`.

Scripts (`datasets/waymo_dataset/`):

- **`waymo_video_extractor.py`** — Generates the `.mp4` video of the sequence's front camera.
- **`waymo_distance_extractor.py`** — Real longitudinal distance to the leading vehicle, frame by frame.
- **`waymo_camera_parameters_extractor.py`** — Camera parameters: intrinsics (`f_u`, `f_v`, `c_u`,
  `c_v`), height, orientation (*yaw*, *pitch*, *roll*) and estimated focal length in mm.
