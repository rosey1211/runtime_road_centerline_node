"""
Export road_best.pth → TorchScript + YAML config for the C++ ROS2 node.

Usage
-----
python export_model.py                          # uses defaults from infer_config.py
python export_model.py --checkpoint road_best.pth --out road_model
"""

import argparse
import os
import sys

import torch
import yaml

# model.py lives two levels up: runtime_version/scripts/ → runtime_version/ → v2/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from model import RoadCNN


def export(checkpoint_path: str, out_stem: str) -> None:
    pt_path   = out_stem + ".pt"
    yaml_path = out_stem + "_config.yaml"

    print(f"Loading checkpoint : {checkpoint_path}")
    model, _, epoch = RoadCNN.from_checkpoint(checkpoint_path, device=torch.device("cpu"))
    model.eval()

    # Trace with a dummy input (batch=1, RGB, H×W)
    dummy = torch.zeros(1, 3, model.image_height, model.image_width)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(pt_path)
    print(f"Saved TorchScript  : {pt_path}")

    # Dump geometry config so the C++ node can post-process without the Python stack
    config = {
        "image_width":         model.image_width,
        "image_height":        model.image_height,
        "n_buckets":           model.n_buckets,
        "n_rows":              3,
        "row_fractions":       [float(f) for f in model.row_fractions],
        "norm_mean":           [float(v) for v in model.norm_mean],
        "norm_std":            [float(v) for v in model.norm_std],
        "crop_top":            float(model.crop_top),
        "crop_bottom":         float(model.crop_bottom),
        "row_bucket_margins":  [float(m) for m in model.row_bucket_margins],
        # bucket_edges[row][edge_idx] — N_ROWS x (N_BUCKETS+1) fractions
        "bucket_edges":        model.bucket_edges_all.tolist(),
        "epoch":               int(epoch),
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved model config : {yaml_path}")
    print(f"  image size  : {model.image_width} x {model.image_height}")
    print(f"  n_buckets   : {model.n_buckets}")
    print(f"  row fracs   : {config['row_fractions']}")
    print(f"  crop        : top={config['crop_top']:.3f}  bottom={config['crop_bottom']:.3f}")
    print("Export complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "road_best.pth"))
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "road_model"))
    args = p.parse_args()
    export(args.checkpoint, args.out)
