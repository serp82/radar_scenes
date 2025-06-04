import sys
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QLabel, QSlider
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from radar_scenes.sequence import Sequence, get_training_sequences
from radar_scenes.labels import ClassificationLabel

from utils import RadarScenesDataset, radar_collate_fn_with_mask
from utils import visualize_radar_batch

CLASS_COLORS = {
    0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'purple', 5: 'gray', -1: 'black'
}

CLASS_NAMES = {
    0: 'Car', 1: 'Pedestrian', 2: 'Pedestrian Group',
    3: 'Two-wheeler', 4: 'Large Vehicle', 5: 'Static', -1: 'Padding'
}


def get_frame_index(sequence_path, target_timestamp):
    with open(sequence_path, 'r') as f:
        data = json.load(f)
    scenes = data["scenes"]
    sorted_ts = sorted([int(ts) for ts in scenes.keys()])
    return sorted_ts.index(target_timestamp) if target_timestamp in sorted_ts else -1


class RadarViewer(QWidget):
    def __init__(self, batch, path_to_dataset):
        super().__init__()
        self.batch = batch
        self.dataset_path = path_to_dataset
        self.idx = 0
        self.max_idx = len(batch["features"]) - 1

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Radar Point Cloud Viewer")
        self.setGeometry(100, 100, 800, 800)

        self.label = QLabel(f"Sample: {self.idx}/{self.max_idx}")
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)

        # 슬라이더 추가
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.max_idx)
        self.slider.setValue(self.idx)
        self.slider.valueChanged.connect(self.slider_changed)

        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_prev.clicked.connect(self.prev_sample)
        self.btn_next.clicked.connect(self.next_sample)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.btn_next)
        self.setLayout(layout)

        self.plot_sample()

    def plot_sample(self):
        self.ax.clear()

        features = self.batch["features"][self.idx]
        labels = self.batch["labels"][self.idx]
        mask = self.batch["mask"][self.idx]
        sequence_name = self.batch["sequence_names"][self.idx]
        timestamps = self.batch["timestamps"][self.idx]

        x = features[:, 0][mask].cpu().numpy()
        y = features[:, 1][mask].cpu().numpy()
        labels = labels[mask].cpu().numpy()

        scene_path = os.path.join(self.dataset_path, "data", sequence_name, "scenes.json")
        ts_strs = []
        for ts in timestamps:
            idx = get_frame_index(scene_path, int(ts))
            ts_strs.append(f"{ts} ({idx})")

        title_str = f"Radar Point Cloud - Seq: {sequence_name}\nTimestamps: " + ", ".join(ts_strs)
        self.ax.set_title(title_str)

        for class_id in np.unique(labels):
            mask_class = labels == class_id
            self.ax.scatter(
                y[mask_class], x[mask_class],
                s=10,
                color=CLASS_COLORS.get(class_id, 'black'),
                label=CLASS_NAMES.get(class_id, f"Unknown {class_id}"),
                alpha=0.6
            )

        self.ax.plot(0, 0, marker='*', color='black', markersize=15, label='Ego Vehicle')
        self.ax.set_xlabel("y (m)")
        self.ax.set_ylabel("x (m)")
        self.ax.axis("equal")
        self.ax.set_xlim(125, -125)
        self.ax.set_ylim(-100, 150)
        self.ax.grid(True)
        self.ax.legend()

        self.label.setText(f"Sample: {self.idx}/{self.max_idx}")
        self.canvas.draw()

    def slider_changed(self, value):
        self.idx = value
        self.plot_sample()

    def prev_sample(self):
        if self.idx > 0:
            self.idx -= 1
            self.slider.setValue(self.idx)  # 슬라이더 값도 갱신됨

    def next_sample(self):
        if self.idx < self.max_idx:
            self.idx += 1
            self.slider.setValue(self.idx)  # 슬라이더 값도 갱신됨


if __name__ == "__main__":

    app = QApplication(sys.argv)
    path_to_dataset = "/home2/Jaehyeok/4_Automotive_Radar/RadarScenes"
    sequence_file = os.path.join(path_to_dataset, "data", "sequences.json")
    training_sequences = get_training_sequences(sequence_file)[:1]  # 샘플 수 적게
    dataset = RadarScenesDataset(training_sequences, path_to_dataset, return_track_ids=False)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=radar_collate_fn_with_mask)

    for batch in dataloader:
        viewer = RadarViewer(batch, path_to_dataset)
        viewer.show()
        sys.exit(app.exec_())
        break