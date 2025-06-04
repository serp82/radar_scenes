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

# --------------------------------------
# main 함수
# --------------------------------------
def main():
    # ✏️ 이 부분 경로에 맞게 수정하세요
    path_to_dataset = "/home2/Jaehyeok/4_Automotive_Radar/RadarScenes"
    sequence_file = os.path.join(path_to_dataset, "data", "sequences.json")

    if not os.path.exists(sequence_file):
        print("❗ sequences.json 경로가 올바르지 않습니다.")
        return

    training_sequences = get_training_sequences(sequence_file)[:1]  # 샘플 수 적게
    dataset = RadarScenesDataset(training_sequences, path_to_dataset, return_track_ids=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=radar_collate_fn_with_mask)

    for batch in dataloader:
        visualize_radar_batch(batch, sample_idx=0, path_to_dataset=path_to_dataset )
        break  # 한 배치만 시각화


if __name__ == "__main__":
    main()