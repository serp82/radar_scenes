import matplotlib.pyplot as plt
import numpy as np
import torch
import os
# 클래스별 색상 정의 (총 6개 클래스: CAR, PEDESTRIAN, ... STATIC)
CLASS_COLORS = {
    0: 'red',       # CAR
    1: 'green',     # PEDESTRIAN
    2: 'blue',      # PEDESTRIAN_GROUP
    3: 'orange',    # TWO_WHEELER
    4: 'purple',    # LARGE_VEHICLE
    5: 'gray',      # STATIC
    -1: 'black'     # padding 무시용
}

CLASS_NAMES = {
    0: 'Car',
    1: 'Pedestrian',
    2: 'Pedestrian Group',
    3: 'Two-wheeler',
    4: 'Large Vehicle',
    5: 'Static',
    -1: 'Padding'
}

import json

def get_frame_index(sequence_path, target_timestamp):
    with open(sequence_path, 'r') as f:
        data = json.load(f)
    scenes = data["scenes"]
    sorted_ts = sorted([int(ts) for ts in scenes.keys()])
    return sorted_ts.index(target_timestamp) if target_timestamp in sorted_ts else -1


def visualize_radar_batch(batch, sample_idx=0, path_to_dataset=None):
    features = batch["features"][sample_idx]
    labels = batch["labels"][sample_idx]
    mask = batch["mask"][sample_idx]
    sequence_name = batch["sequence_names"][sample_idx]
    timestamps = batch["timestamps"][sample_idx]  # list of 4

    x = features[:, 0][mask].cpu().numpy()
    y = features[:, 1][mask].cpu().numpy()
    labels = labels[mask].cpu().numpy()

    # frame index 구하기
    ts_strs = []
    if path_to_dataset is not None:
        scene_path = os.path.join(path_to_dataset, "data", sequence_name, "scenes.json")
        for ts in timestamps:
            idx = get_frame_index(scene_path, int(ts))
            ts_strs.append(f"{ts} ({idx})")
     
    else:
        ts_strs = [str(ts) for ts in timestamps]

    title_str = f"Radar Point Cloud - Seq: {sequence_name}\nTimestamps: " + ", ".join(ts_strs)

    # 시각화
    plt.figure(figsize=(8, 8))
    for class_id in np.unique(labels):
        class_mask = labels == class_id
        plt.scatter(y[class_mask], (x[class_mask]),
                    s=10, label=CLASS_NAMES.get(class_id, f"Unknown {class_id}"),
                    color=CLASS_COLORS.get(class_id, "black"), alpha=0.6)

    plt.plot(0, 0, marker='*', color='black', markersize=15, label="Ego Vehicle")
    plt.xlabel("y (m)")
    plt.ylabel("x (m)")
    plt.title(title_str)
    plt.axis("equal")

    plt.xlim(125, -125)
    plt.ylim(-100, 150)

    plt.legend()
    plt.grid(True)
    plt.show()


