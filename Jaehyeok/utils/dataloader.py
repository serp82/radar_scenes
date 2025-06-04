import os
import numpy as np
import torch
from torch.utils.data import Dataset
from radar_scenes.sequence import Sequence
from radar_scenes.labels import ClassificationLabel


class RadarScenesDataset(Dataset):
    def __init__(self, sequence_names, path_to_dataset, return_track_ids=False, scenes_per_sequence=5, stride=1):
        self.samples = []
        self.return_track_ids = return_track_ids

        for sequence_name in sequence_names:
            try:
                sequence = Sequence.from_json(os.path.join(path_to_dataset, "data", sequence_name, "scenes.json"))
            except FileNotFoundError:
                continue


            timestamps = sorted(sequence.timestamps)

            for i in range(0, len(timestamps) - 3, stride):
                merged_X = []
                merged_y = []
                merged_track_ids = []
                merged_timestamps = []

                for j in range(4):
                    t = timestamps[i + j]
                    scene = sequence.get_scene(t)
                    radar_data = scene.radar_data

                    y_true = np.array([
                        ClassificationLabel.label_to_clabel(x) for x in radar_data["label_id"]
                    ])
                    valid_points = y_true != None
                    y_true = y_true[valid_points]
                    y_true = np.array([x.value for x in y_true], dtype=np.int64)

                    X = np.zeros((len(y_true), 4), dtype=np.float32)
                    X[:, 0] = radar_data["x_cc"][valid_points]
                    X[:, 1] = radar_data["y_cc"][valid_points]
                    X[:, 2] = radar_data["vr_compensated"][valid_points]
                    X[:, 3] = radar_data["rcs"][valid_points]

                    if len(X) == 0:
                        continue  # 비어 있으면 skip

                    merged_X.append(X)
                    merged_y.append(y_true)
                    merged_timestamps.append(t)

                    if self.return_track_ids:
                        merged_track_ids.append(radar_data["track_id"][valid_points])

                if merged_X and merged_y:
                    sample = {
                        "features": torch.from_numpy(np.concatenate(merged_X, axis=0)),
                        "labels": torch.from_numpy(np.concatenate(merged_y, axis=0)),
                        "sequence_name": sequence_name,
                        "timestamps": merged_timestamps  # 4개 timestamp 리스트
                    }

                    if self.return_track_ids:
                        sample["track_ids"] = np.concatenate(merged_track_ids, axis=0)

                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


import torch
from torch.nn.utils.rnn import pad_sequence

def radar_collate_fn_with_mask(batch):
    features = [item["features"] for item in batch]  # List[Tensor(N_i, 4)]
    labels = [item["labels"] for item in batch]      # List[Tensor(N_i)]
    sequence_names = [item["sequence_name"] for item in batch]
    timestamps = [item["timestamps"] for item in batch]

    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)    # (B, max_len, 4)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)         # (B, max_len)

    lengths = torch.tensor([len(l) for l in labels])
    max_len = padded_labels.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)  # (B, max_len)

    batch_data = {
        "features": padded_features,
        "labels": padded_labels,
        "mask": mask,
        "sequence_names": sequence_names,
        "timestamps": timestamps
    }

    if "track_ids" in batch[0]:
        batch_data["track_ids"] = [item["track_ids"] for item in batch]

    return batch_data
