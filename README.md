# road_centerline — ROS2 C++ inference node

Subscribes to a camera image topic, runs the road centerline CNN via LibTorch,
and publishes per-row peak positions on a `CenterlineResult` topic.

## Message: `road_centerline/CenterlineResult`

| Field | Type | Description |
|-------|------|-------------|
| `header` | `std_msgs/Header` | Stamp copied from input image |
| `road_present` | `bool` | True when road classifier fired and confidence passed |
| `road_confidence` | `float32` | Classifier softmax probability for "road" |
| `peak_confidence` | `float32` | Mean per-row peak confidence |
| `row_confidences` | `float32[3]` | Per-row peak confidence [r0=far, r1=mid, r2=near] |
| `points` | `geometry_msgs/Point32[]` | Centerline samples, nearest first |

Each point: `x` = normalised column [0,1], `y` = normalised row [0,1], `z` = confidence.

---

## Step 1 — Export the model

Run once after training (requires the Python environment with PyTorch):

```bash
cd road_code/v2/runtime_version
python scripts/export_model.py \
    --checkpoint ../road_best.pth \
    --out road_model
```

This produces:
- `road_model.pt` — TorchScript model (load with LibTorch)
- `road_model_config.yaml` — geometry / normalisation parameters

---

## Step 2 — Install LibTorch

### Desktop (x86)
```bash
# Get the cxx11 ABI pre-built zip from pytorch.org and unzip it, then:
export TORCH_DIR=/path/to/libtorch/share/cmake/Torch
```

### Jetson Orin Nano (JetPack 6 / PyTorch for Jetson)
```bash
pip3 install torch  # from Nvidia Jetson wheels
export TORCH_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
```

---

## Step 3 — Build the ROS2 package

```bash
# Place this package inside your ROS2 workspace src/
cp -r road_code/v2/runtime_version ~/ros2_ws/src/road_centerline

cd ~/ros2_ws
cmake -DTORCH_DIR=$TORCH_DIR   # or set in colcon env
colcon build --packages-select road_centerline
source install/setup.bash
```

---

## Step 4 — Run

```bash
ros2 run road_centerline road_centerline_node \
    --ros-args \
    -p model_path:=/path/to/road_model.pt \
    -p config_path:=/path/to/road_model_config.yaml \
    -p image_topic:=/camera/image_raw \
    -p output_topic:=road_centerline \
    -p road_threshold:=0.5 \
    -p min_peak_conf:=0.2 \
    -p cluster_thresh:=0.45
```

### Monitor output
```bash
ros2 topic echo /road_centerline
```

---

## Tuning parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `road_threshold` | 0.5 | Min classifier probability to declare road present |
| `min_peak_conf` | 0.2 | Min mean peak confidence across rows |
| `cluster_thresh` | 0.45 | Min fraction of row peak for a bucket to join a cluster |
