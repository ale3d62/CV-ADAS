# Real-time distance detection system using dashcams

Source code of the Bachelor's Thesis *"Real-time distance detection system using dashcams"*.
The full thesis is openly available in the University of Cádiz institutional repository (RODIN):
[hdl.handle.net/10498/33602](http://hdl.handle.net/10498/33602).

Using a single dashcam as its only sensor, connected to a compact on-board computer, the system is an
Advanced Driver-Assistance System (ADAS) based on computer vision. It detects the lane lines and the
surrounding vehicles, estimates the distance to the leading vehicle from geometric calculations based on
the standard lane width, and compares that distance across frames to obtain the relative speed. From the
relative speed it determines whether the safe following distance is being kept and, when it is not, it
emits an audible alert to warn the driver.

The system is provided in two independent, self-contained variants for the perception stage: a **classic
computer vision** pipeline (lane detection via the Hough transform) and a **deep learning** pipeline
(object-detection and multi-task models). Both were designed to run in real time on low-cost embedded
hardware, and the repository also includes the benchmarking code used to measure their accuracy and their
performance on the target devices.

![preview](https://github.com/ale3d62/CV-ADAS/blob/TFG/readmegif.gif?raw=true)

---

## 1. Repository structure

| Folder | Contents |
|---|---|
| `project_classic_cv/` | Classic computer vision variant. Full runnable system with its own `main.py`; lane detection based on image preprocessing and the Hough transform (`lane_finder.py`), vehicle detection and tracking (`car_finder.py`), visualization (`frame_visualizer.py`) and helpers (`auxFunctions.py`). |
| `project_deep_learning/` | Deep learning variant. Full runnable system with its own `main.py`; combined lane and vehicle detection through detection / multi-task models (`lane_car_finder.py`, `car_finder_ncnn.py` for the NCNN backend), visualization and helpers. |
| `models/` | Detection and multi-task models used by both variants, plus the conversion utilities under `utils/` and a README with the download / export instructions. |
| `benchmark/` | Scripts to evaluate detection accuracy against public datasets (`car_benchmark.py`, `lane_benchmark.py`), the YOLOPv2 reference (`yolopv2.py`) and shared utilities. `datasets/` documents the datasets used (BDD100K, TuSimple). |
| `test_videos/` | Input videos on which the system is run, extracted from the [DashCamTours YouTube channel](https://www.youtube.com/@DashCamTours). |

---

## 2. Requirements and installation

The system was developed and tested with **Python 3.7.9**. More recent versions may work, but this is
not guaranteed.

Each variant has its own `requirements.txt`. Using a separate virtual environment per variant is
recommended:

```bash
# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install the dependencies of the variant you want to run
pip install -r project_classic_cv/requirements.txt      # classic computer vision
# or
pip install -r project_deep_learning/requirements.txt   # deep learning
```

The models are not shipped in the repository. Follow `models/README.md` to download the detection model
(e.g. `yolov8n.pt`) and the multi-task model (`v4_2_tasks.onnx`), and place them in `models/`.

---

## 3. Usage

Each variant is launched from its own folder:

```bash
cd project_classic_cv          # or project_deep_learning
python main.py
```

Configuration is done by editing the parameter block at the top of `main.py`. The main options are:

| Parameter | Description |
|---|---|
| `videoSource` | Image source: `video` (test videos in `videoPath`), `screen` (screen capture) or `camera` (device camera). |
| `videoPath` / `inputVideos` | Folder and list of test videos to process when `videoSource = "video"`. |
| `modelName` / `modelPath` | Detection / multi-task model and its folder. |
| `yoloConfThresh`, `yoloIouThresh` | Confidence and IoU thresholds for the detector. |
| `trackingIouThresh` | IoU threshold used to match detections across frames when tracking. |
| `bBoxMinSize` | Minimum bounding-box size (fraction of the image); smaller boxes are ignored. |
| `acceptedClasses` | Object classes to keep (classic variant). |
| `resScaling` | Down-scaling factor `(0, 1]` applied to the frame to run faster (classic variant). |
| `camParams` | Camera parameters `fReal`, `fEq` and the reference `roadWidth` (m) used for the geometric distance estimation. |
| `frameTimeThreshold` / `distanceDiffThreshold` | Time (ms) / distance (m) intervals used to compute the relative speed. |
| `vehiclesDeceleration`, `slowUserBrake`, `reactionTime`, `reactionAproxVel`, `vehicleBonnetSize` | Parameters of the safe-following-distance model. |
| `visualizationMode` | Output: `none`, `screen` or `server` (web server, configured with `serverParameters`). |
| `showSettings`, `showDistances`, `showSpeed` | Elements to overlay on the visualization (deep learning variant). |

---

## 4. System pipeline

The system processes each video frame through the following stages:

1. **Image source** — frames captured from the dashcam, a screen capture, or read from a test video.
2. **Lane detection** — the lane lines of the ego lane are located, either with the classic computer
   vision pipeline (preprocessing + Hough transform) or with a deep learning model.
3. **Vehicle detection** — surrounding vehicles are detected, tracked across frames, and the vehicle in
   the ego lane is identified.
4. **Distance estimation** — the longitudinal distance to the leading vehicle is estimated from geometry,
   using the standard lane width as a reference.
5. **Relative speed estimation** — the estimated distance is compared across frames to obtain the
   relative speed with respect to the leading vehicle.
6. **Safe-distance estimation and alert** — from the relative speed and a vehicle-deceleration model, the
   system determines whether the safe following distance is being kept. When it is not, it emits an
   audible alert.

---

## 5. Benchmarking and target hardware

The system was designed to run in real time on compact, low-cost computers. The `benchmark/` folder
contains the code used to evaluate the perception stage against public datasets — vehicle detection on
**BDD100K** and lane detection on **TuSimple** — and to compare against the YOLOPv2 reference.

Both perception variants (classic computer vision and deep learning) were benchmarked on embedded
devices, measuring execution time, CPU usage and memory usage:

- **Raspberry Pi 4**
- **Radxa Zero 3W**

---

## 6. Citation

Díaz Gómez, A. (2024). *Sistema de detección de distancia en tiempo real mediante dashcams*
[Bachelor's Thesis, Universidad de Cádiz]. RODIN. http://hdl.handle.net/10498/33602

```bibtex
@thesis{diazgomez2024dashcams,
  author = {Díaz Gómez, Alejandro},
  title  = {Sistema de detección de distancia en tiempo real mediante dashcams},
  school = {Universidad de Cádiz},
  year   = {2024},
  month  = {9},
  type   = {Bachelor's Thesis},
  url    = {http://hdl.handle.net/10498/33602}
}
```
